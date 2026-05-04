"""
Gold Layer — Feature Engineering Pipeline
==========================================
Reads all four Silver tables and produces:

  gold_claim_base.parquet
      One row per claim, all Silver reference data joined.
      Uses real denial_flag from replacement data when present; otherwise builds
      a reproducible synthetic label for legacy data.

  gold_claim_features.parquet
      Adds aggregated/model-ready features, including median-imputed amount
      features while preserving original billed_amount for auditability.

Gold contract
-------------
- Reads from Silver only — never from Bronze or raw CSVs.
- Preserves one row per claim. Cost joins must never duplicate claims.
- Preserves raw billed_amount; imputation is feature-level only.
- Uses provided denial_flag when available.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from src.constants import (
    SILVER_META_COLS,
    COL_DIAG_MISSING,
    COL_PROC_MISSING,
    COL_AMOUNT_MISSING,
    COL_PROC_NO_DIAG,
    COL_DIAG_NO_PROC,
    COL_AMOUNT_IMPUTED,
    COL_AMOUNT_IMPUTATION_STRATEGY,
    COL_COST_MATCH_LEVEL,
    COL_COST_MATCH_ENCODED,
    COL_LABEL_SOURCE,
)

logger = structlog.get_logger(__name__)

# Legacy severity encoding retained so existing tests/UI do not break.
# New ML feature severity_rank captures the proper ordinal scale.
SEVERITY_MAP = {"Low": 1, "Medium": 2, "High": 2}
SEVERITY_RANK_MAP = {"Low": 1, "Medium": 2, "High": 3}

_LABEL_SEED = 42


class GoldFeaturePipeline:
    def __init__(self, silver_dir: Path, gold_dir: Path) -> None:
        self.silver_dir = Path(silver_dir)
        self.gold_dir   = Path(gold_dir)
        self._run_ts    = datetime.now(timezone.utc).isoformat()

    def _load_silver(self, dataset_name: str) -> pd.DataFrame:
        path = self.silver_dir / dataset_name / f"{dataset_name}_silver.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Silver file not found: {path}. Run run_silver.py first."
            )
        df = pd.read_parquet(path)
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

    # ── Cost lookup helpers ────────────────────────────────────────────────────

    def _prepare_cost_tables(self, cost: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build cost lookup tables that are safe for the replacement dataset.

        replacement/cost.csv has two rows per procedure across regions. Joining
        claims on procedure_code alone would duplicate claim rows. We therefore:
          1. prepare a region-level table keyed by procedure_code + region;
          2. prepare a procedure-level fallback table with averaged benchmarks.
        """
        cost = cost.copy()
        cost["procedure_code"] = cost["procedure_code"].astype("string").str.strip().str.upper().astype(str)
        cost["region"] = cost["region"].astype("string").str.strip().str.title().astype(str)
        cost["expected_cost"] = pd.to_numeric(cost["expected_cost"], errors="coerce")
        cost["average_cost"] = pd.to_numeric(cost["average_cost"], errors="coerce")
        if "cost_ratio" not in cost.columns:
            cost["cost_ratio"] = cost["average_cost"] / cost["expected_cost"]
        cost["cost_ratio"] = pd.to_numeric(cost["cost_ratio"], errors="coerce")

        regional = (
            cost.groupby(["procedure_code", "region"], as_index=False)
            .agg(
                regional_expected_cost=("expected_cost", "mean"),
                regional_average_cost=("average_cost", "mean"),
                regional_cost_ratio=("cost_ratio", "mean"),
            )
            .rename(columns={"region": "cost_region"})
        )

        procedure = (
            cost.groupby("procedure_code", as_index=False)
            .agg(
                procedure_expected_cost=("expected_cost", "mean"),
                procedure_average_cost=("average_cost", "mean"),
                procedure_cost_ratio=("cost_ratio", "mean"),
            )
        )
        return regional, procedure

    # ── Step 1: Build base table ───────────────────────────────────────────────

    def _build_base(
        self,
        claims: pd.DataFrame,
        providers: pd.DataFrame,
        diagnosis: pd.DataFrame,
        cost: pd.DataFrame,
    ) -> pd.DataFrame:
        row_count_before = len(claims)

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
        )

        regional_cost, procedure_cost = self._prepare_cost_tables(cost)

        # First attempt an exact regional match using provider location.
        df = df.merge(
            regional_cost,
            left_on=["procedure_code", "location"],
            right_on=["procedure_code", "cost_region"],
            how="left",
        )

        # Then attach procedure-level fallback. This table has one row per procedure.
        df = df.merge(procedure_cost, on="procedure_code", how="left")

        df["expected_cost"] = df["regional_expected_cost"].combine_first(df["procedure_expected_cost"])
        df["average_cost"] = df["regional_average_cost"].combine_first(df["procedure_average_cost"])
        df["cost_ratio"] = df["regional_cost_ratio"].combine_first(df["procedure_cost_ratio"])

        df[COL_COST_MATCH_LEVEL] = np.select(
            [
                df["regional_expected_cost"].notna(),
                df["procedure_expected_cost"].notna(),
            ],
            ["regional", "procedure_avg"],
            default="missing",
        )

        helper_cols = [
            "cost_region",
            "regional_expected_cost", "regional_average_cost", "regional_cost_ratio",
            "procedure_expected_cost", "procedure_average_cost", "procedure_cost_ratio",
        ]
        df = df.drop(columns=[c for c in helper_cols if c in df.columns])

        if len(df) != row_count_before:
            raise ValueError(
                f"Gold base row-count changed from {row_count_before} to {len(df)}. "
                "Cost/reference joins must preserve one row per claim."
            )

        computable = df["billed_amount"].notna() & df["expected_cost"].notna()
        df["billed_deviation_pct"] = np.nan
        df.loc[computable, "billed_deviation_pct"] = (
            (df.loc[computable, "billed_amount"] - df.loc[computable, "expected_cost"])
            / df.loc[computable, "expected_cost"] * 100
        ).round(2)

        logger.info(
            "base_table_built",
            total_rows=len(df),
            regional_cost_matches=int((df[COL_COST_MATCH_LEVEL] == "regional").sum()),
            procedure_cost_fallbacks=int((df[COL_COST_MATCH_LEVEL] == "procedure_avg").sum()),
            missing_cost_matches=int((df[COL_COST_MATCH_LEVEL] == "missing").sum()),
            deviation_computable=int(computable.sum()),
        )
        return df

    # ── Step 2: Target label ───────────────────────────────────────────────────

    def _create_denial_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Use real denial_flag when present; otherwise create synthetic labels.

        Kept with the original method name so existing tests/call sites continue
        to work. Real labels are preferred because the replacement dataset now
        provides supervised ML targets.
        """
        df = df.copy()

        if "denial_flag" in df.columns:
            label = pd.to_numeric(df["denial_flag"], errors="coerce")
            valid = label.notna().all() and set(label.dropna().astype(int).unique()).issubset({0, 1})
            if valid:
                df["denial_flag"] = label.astype(int)
                # No synthetic pre-threshold score exists for provided labels.
                if "denial_risk_score" not in df.columns:
                    df["denial_risk_score"] = np.nan
                df[COL_LABEL_SOURCE] = "provided"
                logger.info(
                    "provided_denial_label_used",
                    denied=int(df["denial_flag"].sum()),
                    approved=int((df["denial_flag"] == 0).sum()),
                    denial_rate_pct=round(float(df["denial_flag"].mean() * 100), 1),
                )
                return df

            logger.warning("invalid_or_partial_denial_flag_falling_back_to_synthetic")

        rng = np.random.default_rng(_LABEL_SEED)
        n   = len(df)

        sig_diag_missing  = df[COL_DIAG_MISSING].astype(float)
        sig_proc_no_diag  = df[COL_PROC_NO_DIAG].astype(float)
        sig_proc_missing  = df[COL_PROC_MISSING].astype(float)
        sig_amt_missing   = df[COL_AMOUNT_MISSING].astype(float)
        sig_diag_no_proc  = df[COL_DIAG_NO_PROC].astype(float)

        dev = df["billed_deviation_pct"].fillna(0)
        sig_deviation = (dev.clip(0, 200) / 200).round(4)

        score = (
            0.25 * sig_diag_missing
            + 0.20 * sig_proc_no_diag
            + 0.18 * sig_proc_missing
            + 0.15 * sig_amt_missing
            + 0.12 * sig_diag_no_proc
            + 0.10 * sig_deviation
        )

        noise = rng.normal(0, 0.10, size=n)
        score = (score + noise).clip(0, 1)

        df["denial_risk_score"] = score.round(4)
        df["denial_flag"]       = (score >= 0.5).astype(int)
        df[COL_LABEL_SOURCE] = "synthetic"

        logger.info(
            "synthetic_denial_label_created",
            denied=int(df["denial_flag"].sum()),
            approved=n - int(df["denial_flag"].sum()),
            denial_rate_pct=round(float(df["denial_flag"].mean() * 100), 1),
        )
        return df

    # ── Step 3: Feature engineering ────────────────────────────────────────────

    def _add_amount_imputation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add median-imputed amount columns without overwriting billed_amount.

        Strategy:
          - keep original billed_amount as-is for audit/debugging;
          - if amount is missing, fill from median billed amount for the procedure;
          - fallback to global median when procedure median is unavailable.
        """
        df = df.copy()
        amount = pd.to_numeric(df["billed_amount"], errors="coerce")
        global_median = amount.median()
        if pd.isna(global_median):
            global_median = 0.0

        procedure_medians = (
            df.assign(_amount=amount)
            .dropna(subset=["_amount"])
            .groupby("procedure_code")["_amount"]
            .median()
        )

        imputed = amount.copy()
        strategy = pd.Series("original", index=df.index, dtype="object")

        missing = imputed.isna()
        proc_fill = df.loc[missing, "procedure_code"].map(procedure_medians)
        imputed.loc[missing] = proc_fill
        strategy.loc[missing & proc_fill.notna()] = "procedure_median"

        remaining = imputed.isna()
        imputed.loc[remaining] = global_median
        strategy.loc[remaining] = "global_median"

        df[COL_AMOUNT_IMPUTED] = imputed.astype(float)
        df[COL_AMOUNT_IMPUTATION_STRATEGY] = strategy
        df["log_billed_amount_imputed"] = np.log1p(df[COL_AMOUNT_IMPUTED].clip(lower=0))

        # Preserve legacy raw log feature for dashboards/tests/audit. This remains
        # null when billed_amount is null.
        df["log_billed_amount"] = np.where(
            amount.notna(),
            np.log1p(amount),
            np.nan,
        )
        return df

    def _build_features(self, base: pd.DataFrame) -> pd.DataFrame:
        df = base.copy()

        flag_cols = [
            COL_DIAG_MISSING, COL_PROC_MISSING, COL_AMOUNT_MISSING,
            COL_PROC_NO_DIAG, COL_DIAG_NO_PROC,
        ]

        prov_agg = (
            df.groupby("provider_id")
            .agg(
                provider_claim_count = ("claim_id", "count"),
                provider_avg_billed  = ("billed_amount", "mean"),
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

        pt_freq = (
            df.groupby("patient_id")["claim_id"]
            .count()
            .rename("patient_claim_count")
            .reset_index()
        )
        df = df.merge(pt_freq, on="patient_id", how="left")

        df["severity_encoded"] = df["severity"].map(SEVERITY_MAP).fillna(0).astype(int)
        df["severity_rank"] = df["severity"].map(SEVERITY_RANK_MAP).fillna(0).astype(int)

        specialty_map = {
            "Neurology":  1,
            "Cardiology": 2,
            "Orthopedic": 3,
            "General":    4,
        }
        df["specialty_encoded"] = df["specialty"].map(specialty_map).fillna(0).astype(int)

        df = self._add_amount_imputation_features(df)

        # Raw deviation is preserved. Imputed deviation is model-ready.
        df["billed_deviation_capped"] = df["billed_deviation_pct"].clip(-100, 500)
        computable_imputed_dev = df[COL_AMOUNT_IMPUTED].notna() & df["expected_cost"].notna()
        df["billed_deviation_imputed_pct"] = np.nan
        df.loc[computable_imputed_dev, "billed_deviation_imputed_pct"] = (
            (df.loc[computable_imputed_dev, COL_AMOUNT_IMPUTED]
             - df.loc[computable_imputed_dev, "expected_cost"])
            / df.loc[computable_imputed_dev, "expected_cost"] * 100
        ).round(2)
        df["billed_deviation_imputed_capped"] = df["billed_deviation_imputed_pct"].clip(-100, 500)

        cost_p75 = df["expected_cost"].quantile(0.75)
        df["is_high_cost"] = (
            df[COL_AMOUNT_IMPUTED].notna() & (df[COL_AMOUNT_IMPUTED] > cost_p75)
        ).astype(int)

        df[COL_COST_MATCH_ENCODED] = df[COL_COST_MATCH_LEVEL].map({
            "missing": 0,
            "procedure_avg": 1,
            "regional": 2,
        }).fillna(0).astype(int)

        logger.info("features_built", rows=len(df))
        return df

    def build_feature_manifest(self) -> list[dict]:
        return [
            {"name": COL_DIAG_MISSING, "type": "bool", "group": "missing_flag", "ml_use": True,
             "description": "True if diagnosis_code was null in source"},
            {"name": COL_PROC_MISSING, "type": "bool", "group": "missing_flag", "ml_use": True,
             "description": "True if procedure_code was null in source"},
            {"name": COL_AMOUNT_MISSING, "type": "bool", "group": "missing_flag", "ml_use": True,
             "description": "True if billed_amount was null in source"},
            {"name": COL_PROC_NO_DIAG, "type": "bool", "group": "business_logic", "ml_use": True,
             "description": "Procedure present but diagnosis absent"},
            {"name": COL_DIAG_NO_PROC, "type": "bool", "group": "business_logic", "ml_use": True,
             "description": "Diagnosis present but procedure absent"},
            {"name": "billed_deviation_imputed_capped", "type": "float", "group": "cost", "ml_use": True,
             "description": "% over/under expected cost using median-imputed amount, capped at [-100, 500]",
             "null_strategy": "safe_median_imputer_in_ml_pipeline"},
            {"name": COL_AMOUNT_IMPUTED, "type": "float", "group": "cost", "ml_use": True,
             "description": "billed_amount filled by procedure median, then global median"},
            {"name": "log_billed_amount_imputed", "type": "float", "group": "cost", "ml_use": True,
             "description": "log1p of median-imputed billed amount"},
            {"name": "is_high_cost", "type": "int", "group": "cost", "ml_use": True,
             "description": "1 if imputed billed amount exceeds 75th percentile expected cost"},
            {"name": COL_COST_MATCH_ENCODED, "type": "int", "group": "cost", "ml_use": True,
             "description": "Cost benchmark quality: missing=0, procedure_avg=1, regional=2"},
            {"name": "provider_claim_count", "type": "int", "group": "provider", "ml_use": True,
             "description": "Total claims submitted by this provider in the dataset"},
            {"name": "provider_violation_rate", "type": "float", "group": "provider", "ml_use": True,
             "description": "Average number of structural flags per claim for this provider"},
            {"name": "patient_claim_count", "type": "int", "group": "patient", "ml_use": True,
             "description": "Total claims submitted by this patient in the dataset"},
            {"name": "severity_rank", "type": "int", "group": "diagnosis", "ml_use": True,
             "description": "Diagnosis severity ordinal rank: Low=1, Medium=2, High=3, missing=0"},
            {"name": "severity_encoded", "type": "int", "group": "diagnosis", "ml_use": False,
             "description": "Legacy severity encoding retained for dashboards/tests"},
            {"name": "specialty_encoded", "type": "int", "group": "provider", "ml_use": True,
             "description": "Provider specialty label-encoded"},
            # Audit/non-ML columns
            {"name": "claim_id", "type": "str", "ml_use": False, "description": "Claim identifier"},
            {"name": "denial_flag", "type": "int", "ml_use": False,
             "description": "Target label: real if available, synthetic for legacy data"},
            {"name": "denial_risk_score", "type": "float", "ml_use": False,
             "description": "Synthetic pre-threshold risk score; null when real label is used"},
            {"name": COL_LABEL_SOURCE, "type": "str", "ml_use": False,
             "description": "provided or synthetic"},
            {"name": "log_billed_amount", "type": "float", "ml_use": False,
             "description": "Raw log1p(billed_amount); stays null when original amount is null"},
            {"name": "billed_deviation_capped", "type": "float", "ml_use": False,
             "description": "Raw billed deviation capped; retained for audit/plots"},
            {"name": COL_AMOUNT_IMPUTATION_STRATEGY, "type": "str", "ml_use": False,
             "description": "original, procedure_median, or global_median"},
            {"name": COL_COST_MATCH_LEVEL, "type": "str", "ml_use": False,
             "description": "regional, procedure_avg, or missing"},
        ]

    def run(self) -> dict:
        report: dict = {"run_timestamp": self._run_ts}

        try:
            claims    = self._load_silver("claims")
            providers = self._load_silver("providers")
            diagnosis = self._load_silver("diagnosis")
            cost      = self._load_silver("cost")

            base = self._build_base(claims, providers, diagnosis, cost)
            base = self._create_denial_label(base)
            base_path = self._write_gold(base, "gold_claim_base")

            features = self._build_features(base)
            feat_path = self._write_gold(features, "gold_claim_features")

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
                "label_source":    str(base[COL_LABEL_SOURCE].iloc[0]) if COL_LABEL_SOURCE in base.columns and not base.empty else "unknown",
                "cost_match_counts": base[COL_COST_MATCH_LEVEL].value_counts(dropna=False).to_dict(),
                "base_path":       str(base_path),
                "feature_path":    str(feat_path),
                "manifest_path":   str(manifest_path),
            })

        except Exception as exc:
            logger.exception("gold_pipeline_failed", error=str(exc))
            report["status"] = "failed"
            report["error"]  = str(exc)

        return report
