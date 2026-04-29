"""
Gold Layer — Feature Engineering Pipeline
==========================================
Reads all four Silver tables and produces two Gold Parquet files:

  gold_claim_base.parquet
      One row per claim, all Silver reference data joined.
      Computes billed_deviation_pct.
      Adds synthetic denial_flag.

  gold_claim_features.parquet
      Adds aggregated features (provider stats, patient frequency).
      Adds encoded categorical columns ready for ML.
      All Gold features documented with build_feature_manifest().

Gold contract
-------------
- Reads from Silver only — never from Bronze or raw CSVs.
- Never writes imputed values back (nulls remain null in the Parquet).
- ML preprocessing (null filling, scaling) happens inside src/ml/train.py.
- The synthetic denial_flag is clearly documented and reproducible.

Synthetic label design
----------------------
We have no real claim_status. The label is built as a weighted risk score:
  - Structural violations (missing fields, business rule flags) → primary signal
  - Cost deviation → secondary signal
  - Gaussian noise added → prevents model from simply memorising the rules
  - Threshold at 0.5 → ~35–38% denial rate (realistic for a messy dataset)

This means the ML model must genuinely learn the patterns, not memorise flag→label.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import structlog

from src.constants import (
    SENTINEL_MISSING,
    SILVER_META_COLS,
    COL_DIAG_MISSING,
    COL_PROC_MISSING,
    COL_AMOUNT_MISSING,
    COL_PROC_NO_DIAG,
    COL_DIAG_NO_PROC,
    CRITICAL_FIELDS,
)

logger = structlog.get_logger(__name__)

# Severity → numeric mapping used across Gold and ML
SEVERITY_MAP = {"High": 2, "Low": 1}

# Synthetic label random seed — fixed so results are reproducible
_LABEL_SEED = 42


class GoldFeaturePipeline:
    """
    Builds Gold base and feature tables from Silver Parquet files.

    Parameters
    ----------
    silver_dir : Root of Silver Parquet subdirectories.
    gold_dir   : Root of Gold Parquet output (created if absent).
    """

    def __init__(self, silver_dir: Path, gold_dir: Path) -> None:
        self.silver_dir = Path(silver_dir)
        self.gold_dir   = Path(gold_dir)
        self._run_ts    = datetime.now(timezone.utc).isoformat()

    # ── File I/O ───────────────────────────────────────────────────────────────

    def _load_silver(self, dataset_name: str) -> pd.DataFrame:
        path = self.silver_dir / dataset_name / f"{dataset_name}_silver.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Silver file not found: {path}. Run run_silver.py first."
            )
        df = pd.read_parquet(path)
        # Drop pipeline metadata columns — Gold only keeps business data
        drop = [c for c in SILVER_META_COLS if c in df.columns]
        df = df.drop(columns=drop)
        logger.info("silver_loaded", dataset=dataset_name, rows=len(df))
        return df

    def _write_gold(self, df: pd.DataFrame, table_name: str) -> Path:
        self.gold_dir.mkdir(parents=True, exist_ok=True)
        path = self.gold_dir / f"{table_name}.parquet"
        df.to_parquet(path, index=False, engine="pyarrow")
        logger.info("gold_written", table=table_name, rows=len(df), path=str(path))
        return path

    # ── Step 1: Build base table ───────────────────────────────────────────────

    def _build_base(
        self,
        claims: pd.DataFrame,
        providers: pd.DataFrame,
        diagnosis: pd.DataFrame,
        cost: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Join all four Silver tables into one fact row per claim.
        Computes billed_deviation_pct using procedure-level expected cost only
        (regional join would only cover 12.6% of claims — documented decision).
        """
        # Left joins so no claims are ever dropped
        df = (
            claims
            .merge(
                providers[["provider_id", "specialty", "location"]],
                on="provider_id", how="left"
            )
            .merge(
                diagnosis[["diagnosis_code", "category", "severity"]],
                on="diagnosis_code", how="left"
            )
            .merge(
                cost[["procedure_code", "expected_cost", "average_cost", "cost_ratio"]],
                on="procedure_code", how="left"
            )
        )

        # Cost deviation: (billed - expected) / expected * 100
        # Null when either billed_amount or expected_cost is missing.
        # NEVER imputed here — ML pipeline handles nulls.
        computable = df["billed_amount"].notna() & df["expected_cost"].notna()
        df["billed_deviation_pct"] = np.nan
        df.loc[computable, "billed_deviation_pct"] = (
            (df.loc[computable, "billed_amount"] - df.loc[computable, "expected_cost"])
            / df.loc[computable, "expected_cost"] * 100
        ).round(2)

        logger.info(
            "base_table_built",
            total_rows=len(df),
            deviation_computable=int(computable.sum()),
        )
        return df

    # ── Step 2: Synthetic denial label ─────────────────────────────────────────

    def _create_denial_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a reproducible synthetic denial_flag.

        Method: weighted risk score → add noise → threshold at 0.5
        The noise prevents the ML model from trivially memorising the rules.

        Risk components (weights sum to 1.0 for a fully-bad claim):
          0.25  diagnosis_code_missing — auto-deny signal
          0.20  proc_no_diag          — procedure without clinical justification
          0.18  procedure_code_missing — nothing billed
          0.15  billed_amount_missing  — no financial record
          0.12  diag_no_proc           — condition documented, nothing billed
          0.10  cost deviation >200%   — extreme overbilling
          noise Gaussian(0, 0.10)     — simulates real-world edge cases

        Threshold 0.50: any 2–3 flags firing together crosses into denial,
        even without the single most dominant flag present.

        Threshold 0.5 → ~35–38% denial rate.

        denial_flag: 1 = predicted denial, 0 = predicted approval
        denial_risk_score: continuous score before thresholding (kept for audit)
        """
        rng = np.random.default_rng(_LABEL_SEED)
        n   = len(df)

        # Each component is a 0-1 signal
        sig_diag_missing  = df[COL_DIAG_MISSING].astype(float)
        sig_proc_no_diag  = df[COL_PROC_NO_DIAG].astype(float)
        sig_proc_missing  = df[COL_PROC_MISSING].astype(float)
        sig_amt_missing   = df[COL_AMOUNT_MISSING].astype(float)
        sig_diag_no_proc  = df[COL_DIAG_NO_PROC].astype(float)

        # Cost deviation signal: clamp to [0, 1] where 1 = ≥200% overbilling
        dev = df["billed_deviation_pct"].fillna(0)
        sig_deviation = (dev.clip(0, 200) / 200).round(4)

        # Weighted sum — all five business logic flags contribute.
        # Weights are balanced so that any 2-3 flags firing together crosses
        # the 0.50 threshold even without the top individual signal.
        score = (
            0.25 * sig_diag_missing   # auto-deny: missing diagnosis
            + 0.20 * sig_proc_no_diag  # billing without clinical justification
            + 0.18 * sig_proc_missing  # no procedure billed at all
            + 0.15 * sig_amt_missing   # no financial record
            + 0.12 * sig_diag_no_proc  # condition documented but nothing billed
            + 0.10 * sig_deviation     # overbilling vs benchmark
        )

        # Add noise: simulates approved claims with flags (edge cases / appeals)
        noise = rng.normal(0, 0.10, size=n)
        score = (score + noise).clip(0, 1)

        df = df.copy()
        df["denial_risk_score"] = score.round(4)
        df["denial_flag"]       = (score >= 0.5).astype(int)

        denied_count = int(df["denial_flag"].sum())
        logger.info(
            "denial_label_created",
            denied=denied_count,
            approved=n - denied_count,
            denial_rate_pct=round(denied_count / n * 100, 1),
        )
        return df

    # ── Step 3: Feature engineering ────────────────────────────────────────────

    def _build_features(self, base: pd.DataFrame) -> pd.DataFrame:
        """
        Add aggregated and derived features to the base table.

        Feature groups:
          Provider-level  — aggregated from all claims per provider
          Patient-level   — claim frequency per patient
          Severity         — encoded numeric severity
          Specialty        — encoded specialty
          Cost             — log-transformed amount, high-cost flag

        All new columns are documented in build_feature_manifest().
        """
        df = base.copy()

        # ── Provider-level aggregations ────────────────────────────────────────
        # Computed across ALL claims (before any split) — represents provider history.
        # This is a training-time decision: in production, these will be precomputed
        # from the full claims history database (Week 7).

        flag_cols = [COL_DIAG_MISSING, COL_PROC_MISSING, COL_AMOUNT_MISSING,
                     COL_PROC_NO_DIAG, COL_DIAG_NO_PROC]

        prov_agg = (
            df.groupby("provider_id")
            .agg(
                provider_claim_count = ("claim_id", "count"),
                provider_avg_billed  = ("billed_amount", "mean"),
                # Avg number of flags per claim — high means risky provider
                provider_violation_rate = (
                    COL_DIAG_MISSING,
                    lambda x: df.loc[x.index, flag_cols].sum(axis=1).mean()
                ),
            )
            .reset_index()
        )
        prov_agg["provider_avg_billed"]     = prov_agg["provider_avg_billed"].round(2)
        prov_agg["provider_violation_rate"] = prov_agg["provider_violation_rate"].round(4)

        df = df.merge(prov_agg, on="provider_id", how="left")

        # ── Patient-level aggregation ──────────────────────────────────────────
        # High repeat-claim frequency can signal duplicate billing
        pt_freq = (
            df.groupby("patient_id")["claim_id"]
            .count()
            .rename("patient_claim_count")
            .reset_index()
        )
        df = df.merge(pt_freq, on="patient_id", how="left")

        # ── Severity encoding ──────────────────────────────────────────────────
        # High=2, Low=1, null (MISSING code) = 0
        df["severity_encoded"] = df["severity"].map(SEVERITY_MAP).fillna(0).astype(int)

        # ── Specialty encoding ─────────────────────────────────────────────────
        # Label-encode. Fixed mapping so train and inference use the same codes.
        # New specialties in production will get 0 (unknown).
        specialty_map = {
            "Neurology":    1,
            "Cardiology":   2,
            "Orthopedic":   3,
            "General":      4,
        }
        df["specialty_encoded"] = df["specialty"].map(specialty_map).fillna(0).astype(int)

        # ── Cost features ──────────────────────────────────────────────────────
        # Log-transform: compresses right-skewed distribution (min ₹633 → max ₹49,869)
        df["log_billed_amount"] = np.where(
            df["billed_amount"].notna(),
            np.log1p(df["billed_amount"]),
            np.nan,
        )

        # High-cost flag: billed_amount above 75th percentile of expected_cost
        cost_p75 = df["expected_cost"].quantile(0.75)
        df["is_high_cost"] = (
            df["billed_amount"].notna() & (df["billed_amount"] > cost_p75)
        ).astype(int)

        # Capped deviation: reduces influence of extreme outliers (4011%)
        # Capped at 500% so PROC6's extreme values don't dominate
        df["billed_deviation_capped"] = df["billed_deviation_pct"].clip(-100, 500)

        logger.info(
            "features_built",
            rows=len(df),
            new_feature_count=5,  # prov_claim_count, prov_violation_rate, pt_count, severity_enc, specialty_enc
        )
        return df

    # ── Feature manifest ───────────────────────────────────────────────────────

    def build_feature_manifest(self) -> list[dict]:
        """
        Returns a list of dicts documenting every Gold feature.
        Written to gold_dir/feature_manifest.json by run().
        Consumed by src/ml/train.py to select ML features.
        """
        return [
            # ── From Silver (flags) ──────────────────────────────────────────
            {
                "name": COL_DIAG_MISSING,
                "type": "bool",
                "group": "missing_flag",
                "ml_use": True,
                "description": "True if diagnosis_code was null in source",
            },
            {
                "name": COL_PROC_MISSING,
                "type": "bool",
                "group": "missing_flag",
                "ml_use": True,
                "description": "True if procedure_code was null in source",
            },
            {
                "name": COL_AMOUNT_MISSING,
                "type": "bool",
                "group": "missing_flag",
                "ml_use": True,
                "description": "True if billed_amount is null",
            },
            {
                "name": COL_PROC_NO_DIAG,
                "type": "bool",
                "group": "business_logic",
                "ml_use": True,
                "description": "Procedure present but diagnosis absent",
            },
            {
                "name": COL_DIAG_NO_PROC,
                "type": "bool",
                "group": "business_logic",
                "ml_use": True,
                "description": "Diagnosis present but procedure absent",
            },
            # ── Cost features ───────────────────────────────────────────────
            {
                "name": "billed_deviation_capped",
                "type": "float",
                "group": "cost",
                "ml_use": True,
                "description": "% overbilling vs expected_cost, capped at [-100, 500]",
                "null_strategy": "fill_median",
            },
            {
                "name": "log_billed_amount",
                "type": "float",
                "group": "cost",
                "ml_use": True,
                "description": "log1p(billed_amount) — compresses right-skewed distribution",
                "null_strategy": "fill_median",
            },
            {
                "name": "is_high_cost",
                "type": "int",
                "group": "cost",
                "ml_use": True,
                "description": "1 if billed_amount > 75th percentile of expected_cost",
            },
            # ── Provider features ───────────────────────────────────────────
            {
                "name": "provider_claim_count",
                "type": "int",
                "group": "provider",
                "ml_use": True,
                "description": "Total claims submitted by this provider in the dataset",
            },
            {
                "name": "provider_violation_rate",
                "type": "float",
                "group": "provider",
                "ml_use": True,
                "description": "Avg number of flags per claim for this provider",
            },
            # ── Patient features ────────────────────────────────────────────
            {
                "name": "patient_claim_count",
                "type": "int",
                "group": "patient",
                "ml_use": True,
                "description": "Total claims submitted by this patient in the dataset",
            },
            # ── Categorical encoded ──────────────────────────────────────────
            {
                "name": "severity_encoded",
                "type": "int",
                "group": "diagnosis",
                "ml_use": True,
                "description": "Diagnosis severity: High=2, Low=1, missing=0",
            },
            {
                "name": "specialty_encoded",
                "type": "int",
                "group": "provider",
                "ml_use": True,
                "description": "Provider specialty label-encoded (Neurology=1, Cardiology=2, Orthopedic=3, General=4, unknown=0)",
            },
            # ── Not used in ML (ID / label / audit) ─────────────────────────
            {
                "name": "claim_id",      "type": "str",   "ml_use": False,
                "description": "Claim identifier",
            },
            {
                "name": "denial_flag",   "type": "int",   "ml_use": False,
                "description": "Synthetic target: 1=denied, 0=approved",
            },
            {
                "name": "denial_risk_score", "type": "float", "ml_use": False,
                "description": "Continuous pre-threshold score used to create denial_flag",
            },
        ]

    # ── Orchestrator ───────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Full Gold pipeline: Silver → base table → features → Parquet.

        Returns
        -------
        dict with run_timestamp, table paths, row counts, denial rate.
        """
        report: dict = {"run_timestamp": self._run_ts}

        try:
            # Load all Silver tables
            claims    = self._load_silver("claims")
            providers = self._load_silver("providers")
            diagnosis = self._load_silver("diagnosis")
            cost      = self._load_silver("cost")

            # Build base table
            base = self._build_base(claims, providers, diagnosis, cost)
            base = self._create_denial_label(base)
            base_path = self._write_gold(base, "gold_claim_base")

            # Build feature table
            features = self._build_features(base)
            feat_path = self._write_gold(features, "gold_claim_features")

            # Write feature manifest
            import json
            manifest = self.build_feature_manifest()
            manifest_path = self.gold_dir / "feature_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            logger.info("feature_manifest_written", path=str(manifest_path))

            report.update({
                "status":          "success",
                "base_rows":       len(base),
                "feature_rows":    len(features),
                "denied_count":    int(base["denial_flag"].sum()),
                "denial_rate_pct": round(base["denial_flag"].mean() * 100, 1),
                "base_path":       str(base_path),
                "feature_path":    str(feat_path),
                "manifest_path":   str(manifest_path),
            })

        except Exception as exc:
            logger.exception("gold_pipeline_failed", error=str(exc))
            report["status"] = "failed"
            report["error"]  = str(exc)

        return report
