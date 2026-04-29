"""
ML Training Pipeline
======================
Reads gold_claim_features.parquet → trains two models → evaluates → saves.

Models trained:
  1. Logistic Regression — interpretable baseline
  2. XGBoost — main model (better handles mixed feature types)

Preprocessing (inside pipeline, never written to Gold):
  - Median imputation for float nulls (billed_deviation_capped, log_billed_amount)
  - StandardScaler for Logistic Regression
  - XGBoost handles missing values natively — no imputation needed

Evaluation metrics (all on held-out test set):
  - ROC-AUC  (primary ranking metric)
  - Recall   (most important: missing a real denial is worse than a false alarm)
  - Precision
  - F1
  - Accuracy

Saved artefacts (models/ directory):
  - lr_model.pkl           — fitted LogisticRegression + preprocessing pipeline
  - xgb_model.pkl          — fitted XGBoost + preprocessing pipeline
  - training_report.json   — all metrics, feature list, threshold used
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import structlog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb

logger = structlog.get_logger(__name__)

# These must match exactly what GoldFeaturePipeline creates
# Loaded from feature_manifest.json at runtime — this is the fallback
_ML_FEATURES: list[str] = [
    "diagnosis_code_missing",
    "procedure_code_missing",
    "billed_amount_missing",
    "proc_no_diag",
    "diag_no_proc",
    "billed_deviation_capped",
    "log_billed_amount",
    "is_high_cost",
    "provider_claim_count",
    "provider_violation_rate",
    "patient_claim_count",
    "severity_encoded",
    "specialty_encoded",
]

_TARGET   = "denial_flag"
_TEST_SIZE = 0.30
_SEED      = 42


def _load_feature_list(manifest_path: Optional[Path]) -> list[str]:
    """
    Load ML feature names from the feature manifest if it exists.
    Falls back to _ML_FEATURES if the manifest is missing.
    """
    if manifest_path and manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        features = [m["name"] for m in manifest if m.get("ml_use") is True]
        logger.info("feature_list_loaded_from_manifest", count=len(features))
        return features
    logger.warning("feature_manifest_not_found_using_default")
    return _ML_FEATURES


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute all evaluation metrics. Returns a plain dict (JSON-serialisable)."""
    return {
        "roc_auc":   round(float(roc_auc_score(y_true, y_prob)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "support": {
            "denied":   int(y_true.sum()),
            "approved": int((y_true == 0).sum()),
            "total":    len(y_true),
        },
    }


class ModelTrainer:
    """
    Full training pipeline: Gold features → trained models → saved artefacts.

    Parameters
    ----------
    gold_dir   : Directory containing gold_claim_features.parquet and feature_manifest.json.
    models_dir : Where trained models and reports are saved.
    """

    def __init__(self, gold_dir: Path, models_dir: Path) -> None:
        self.gold_dir   = Path(gold_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _load_features(self) -> pd.DataFrame:
        path = self.gold_dir / "gold_claim_features.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Gold feature table not found: {path}. Run run_gold.py first."
            )
        df = pd.read_parquet(path)
        logger.info("gold_features_loaded", rows=len(df), cols=len(df.columns))
        return df

    def _prepare_data(
        self,
        df: pd.DataFrame,
        features: list[str],
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Select features, validate target, stratified split.
        Returns (X_train, X_test, y_train, y_test).
        """
        # Validate features exist
        missing_cols = [f for f in features if f not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Features missing from Gold table: {missing_cols}. "
                f"Re-run run_gold.py to regenerate."
            )

        X = df[features].copy()
        y = df[_TARGET].astype(int)

        # Cast bool columns to int (sklearn handles ints better than bool)
        bool_cols = X.select_dtypes(include="bool").columns
        X[bool_cols] = X[bool_cols].astype(int)

        # Stratified split: preserves class ratio in both sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=_TEST_SIZE, random_state=_SEED, stratify=y
        )

        logger.info(
            "data_split",
            train_rows=len(X_train),
            test_rows=len(X_test),
            train_denied=int(y_train.sum()),
            test_denied=int(y_test.sum()),
        )
        return X_train, X_test, y_train, y_test

    def _train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Pipeline:
        """
        Logistic Regression with median imputation + StandardScaler.
        Imputation fitted on train set only to prevent data leakage.
        """
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   LogisticRegression(
                max_iter=500,
                random_state=_SEED,
                class_weight="balanced",   # handles class imbalance
                C=1.0,
            )),
        ])
        pipeline.fit(X_train, y_train)
        logger.info("logistic_regression_trained")
        return pipeline

    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Pipeline:
        """
        XGBoost with median imputation only (no scaling needed for trees).
        XGBoost handles remaining nulls natively.
        scale_pos_weight adjusts for class imbalance.
        """
        n_neg    = int((y_train == 0).sum())
        n_pos    = int((y_train == 1).sum())
        pos_weight = n_neg / max(n_pos, 1)  # ~1.7 for our data

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model",   xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=pos_weight,
                random_state=_SEED,
                eval_metric="logloss",
                verbosity=0,
            )),
        ])
        pipeline.fit(X_train, y_train)
        logger.info(
            "xgboost_trained",
            n_estimators=200,
            scale_pos_weight=round(pos_weight, 3),
        )
        return pipeline

    def _compute_shap_importances(
        self,
        model_pipeline: Pipeline,
        X_test: pd.DataFrame,
        feature_names: list[str],
        n_samples: int = 200,
    ) -> dict[str, float]:
        """
        Compute mean |SHAP value| per feature on a sample of test rows.
        Returns feature_name → mean absolute SHAP value (sorted descending).
        Used downstream by predict.py to explain individual predictions.
        """
        import shap

        # Apply imputer (same pipeline step) before SHAP
        X_imputed = model_pipeline.named_steps["imputer"].transform(X_test)
        X_sample  = X_imputed[:n_samples]

        xgb_model = model_pipeline.named_steps["model"]
        explainer  = shap.TreeExplainer(xgb_model)
        shap_vals  = explainer.shap_values(X_sample)

        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        importance_map = {
            name: round(float(val), 6)
            for name, val in zip(feature_names, mean_abs_shap)
        }
        # Sort by importance descending
        return dict(sorted(importance_map.items(), key=lambda x: -x[1]))

    def run(self) -> dict:
        """
        Full training run. Trains both models, evaluates on test set,
        saves models + training report. Returns a structured report dict.
        """
        import pickle

        report: dict = {}

        try:
            df       = self._load_features()
            features = _load_feature_list(self.gold_dir / "feature_manifest.json")
            X_train, X_test, y_train, y_test = self._prepare_data(df, features)

            # ── Logistic Regression ──────────────────────────────────────────
            lr_pipeline = self._train_logistic_regression(X_train, y_train)
            lr_pred  = lr_pipeline.predict(X_test)
            lr_prob  = lr_pipeline.predict_proba(X_test)[:, 1]
            lr_metrics = _compute_metrics(y_test.values, lr_pred, lr_prob)
            logger.info("lr_evaluation", **lr_metrics)

            # ── XGBoost ──────────────────────────────────────────────────────
            xgb_pipeline = self._train_xgboost(X_train, y_train)
            xgb_pred  = xgb_pipeline.predict(X_test)
            xgb_prob  = xgb_pipeline.predict_proba(X_test)[:, 1]
            xgb_metrics = _compute_metrics(y_test.values, xgb_pred, xgb_prob)
            logger.info("xgb_evaluation", **xgb_metrics)

            # ── SHAP importances (XGBoost only) ──────────────────────────────
            shap_importance = self._compute_shap_importances(xgb_pipeline, X_test, features)
            top5_shap = dict(list(shap_importance.items())[:5])
            logger.info("shap_computed", top5=top5_shap)

            # ── Save models ───────────────────────────────────────────────────
            lr_path  = self.models_dir / "lr_model.pkl"
            xgb_path = self.models_dir / "xgb_model.pkl"

            with open(lr_path, "wb") as f:
                pickle.dump({"pipeline": lr_pipeline, "features": features}, f)
            with open(xgb_path, "wb") as f:
                pickle.dump({"pipeline": xgb_pipeline, "features": features}, f)

            logger.info("models_saved", lr=str(lr_path), xgb=str(xgb_path))

            # ── Training report ────────────────────────────────────────────────
            training_report = {
                "features":          features,
                "feature_count":     len(features),
                "target":            _TARGET,
                "train_rows":        len(X_train),
                "test_rows":         len(X_test),
                "test_denied":       int(y_test.sum()),
                "test_approved":     int((y_test == 0).sum()),
                "logistic_regression": lr_metrics,
                "xgboost":             xgb_metrics,
                "shap_importance":     shap_importance,
                "recommended_model":   (
                    "xgboost"
                    if xgb_metrics["roc_auc"] >= lr_metrics["roc_auc"]
                    else "logistic_regression"
                ),
            }
            report_path = self.models_dir / "training_report.json"
            with open(report_path, "w") as f:
                json.dump(training_report, f, indent=2)
            logger.info("training_report_saved", path=str(report_path))

            report.update({
                "status":          "success",
                "recommended":     training_report["recommended_model"],
                "lr_roc_auc":      lr_metrics["roc_auc"],
                "xgb_roc_auc":     xgb_metrics["roc_auc"],
                "xgb_recall":      xgb_metrics["recall"],
                "lr_path":         str(lr_path),
                "xgb_path":        str(xgb_path),
                "report_path":     str(report_path),
            })

        except Exception as exc:
            logger.exception("training_failed", error=str(exc))
            report["status"] = "failed"
            report["error"]  = str(exc)

        return report
