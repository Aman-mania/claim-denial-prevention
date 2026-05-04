"""
ML Training Pipeline
====================
Reads gold_claim_features.parquet → trains candidate models → tunes the
business threshold → saves models, threshold report, calibration report, and a
model card.

Design goals
------------
- Keep Week 4 local-first and simple to run.
- Treat denial prevention as a recall-first business problem.
- Avoid hard-coded 0.50 decisions by tuning a threshold on validation data.
- Preserve cloud migration readiness: all important metadata is saved as JSON
  artifacts that can later be logged to MLflow / Databricks Model Registry.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import structlog
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

logger = structlog.get_logger(__name__)

# Fallback feature list. In normal runs, feature_manifest.json is the source of truth.
_ML_FEATURES: list[str] = [
    "diagnosis_code_missing",
    "procedure_code_missing",
    "billed_amount_missing",
    "proc_no_diag",
    "diag_no_proc",
    "billed_deviation_imputed_capped",
    "billed_amount_imputed",
    "log_billed_amount_imputed",
    "is_high_cost",
    "cost_match_encoded",
    "provider_claim_count",
    "provider_violation_rate",
    "patient_claim_count",
    "severity_rank",
    "specialty_encoded",
]

_TARGET = "denial_flag"
_SEED = 42

# Split policy: train model on 60%, tune threshold on 20%, report final metrics on 20%.
_TEST_SIZE = 0.20
_VALIDATION_SIZE_OF_REMAINING = 0.25  # 0.25 * 80% = 20%

# Denial prevention should prefer catching likely denials over minimizing reviews.
_MIN_DENIAL_RECALL = 0.90


def _load_feature_list(manifest_path: Optional[Path]) -> list[str]:
    """Load ML feature names from the Gold feature manifest, with a stable fallback."""
    if manifest_path and manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        features = [m["name"] for m in manifest if m.get("ml_use") is True]
        logger.info("feature_list_loaded_from_manifest", count=len(features))
        return features
    logger.warning("feature_manifest_not_found_using_default")
    return _ML_FEATURES


def _class_predictions(prob: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(prob) >= threshold).astype(int)


def _compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> dict:
    """Compute classification metrics at a specific decision threshold."""
    y_pred = _class_predictions(y_prob, threshold)
    return {
        "threshold": round(float(threshold), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "false_negatives": int(((y_true == 1) & (y_pred == 0)).sum()),
        "false_positives": int(((y_true == 0) & (y_pred == 1)).sum()),
        "true_positives": int(((y_true == 1) & (y_pred == 1)).sum()),
        "true_negatives": int(((y_true == 0) & (y_pred == 0)).sum()),
        "support": {
            "denied": int(np.asarray(y_true).sum()),
            "approved": int((np.asarray(y_true) == 0).sum()),
            "total": int(len(y_true)),
        },
    }


def _tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float = _MIN_DENIAL_RECALL,
) -> tuple[dict, list[dict]]:
    """
    Tune a decision threshold on validation probabilities.

    Selection rule:
    1. Prefer thresholds that achieve the target denial recall.
    2. Among those, choose highest F1, then highest precision.
    3. If no threshold reaches target recall, choose highest F2-like score.

    This makes the 49% scenario explicit: near-threshold claims are not silently
    treated as safe; the selected threshold and risk bands decide review action.
    """
    rows: list[dict] = []
    for threshold in np.round(np.linspace(0.05, 0.95, 181), 4):
        m = _compute_metrics(y_true, y_prob, float(threshold))
        beta2 = 2.0
        precision = m["precision"]
        recall = m["recall"]
        denom = beta2**2 * precision + recall
        f2 = 0.0 if denom == 0 else (1 + beta2**2) * precision * recall / denom
        m["f2"] = round(float(f2), 4)
        m["meets_min_recall"] = bool(recall >= min_recall)
        rows.append(m)

    candidates = [r for r in rows if r["meets_min_recall"]]
    if candidates:
        selected = sorted(
            candidates,
            key=lambda r: (r["f1"], r["precision"], r["threshold"]),
            reverse=True,
        )[0]
        selected["selection_reason"] = f"highest F1 among thresholds with recall >= {min_recall:.2f}"
    else:
        selected = sorted(rows, key=lambda r: (r["f2"], r["recall"], r["precision"]), reverse=True)[0]
        selected["selection_reason"] = f"fallback: highest F2 because no threshold reached recall >= {min_recall:.2f}"

    return selected, rows


def _risk_band_policy(classification_threshold: float) -> dict:
    """
    Convert the tuned denial threshold into risk bands.

    - HIGH means the model would classify the claim as likely denied.
    - MEDIUM means the claim is not classified as denied yet, but deserves review.
    - LOW means normal/light validation only.
    """
    high = float(classification_threshold)
    review = float(max(0.20, min(0.40, high * 0.70)))
    return {
        "low_upper_exclusive": round(review, 4),
        "medium_lower_inclusive": round(review, 4),
        "medium_upper_exclusive": round(high, 4),
        "high_lower_inclusive": round(high, 4),
        "classification_threshold": round(high, 4),
        "policy": "LOW below review threshold; MEDIUM between review and tuned denial threshold; HIGH at/above tuned threshold",
    }


def _calibration_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Create a lightweight calibration/reliability report without adding another
    training dependency. Brier score is lower-is-better; ECE is approximate.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    for i in range(n_bins):
        left = bins[i]
        right = bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)

        count = int(mask.sum())
        if count == 0:
            rows.append({
                "bin": i + 1,
                "range": f"[{left:.1f}, {right:.1f}{']' if i == n_bins - 1 else ')'}",
                "count": 0,
                "avg_predicted_probability": None,
                "actual_denial_rate": None,
                "absolute_gap": None,
            })
            continue

        avg_prob = float(y_prob[mask].mean())
        actual_rate = float(y_true[mask].mean())
        gap = abs(avg_prob - actual_rate)
        ece += (count / len(y_true)) * gap
        rows.append({
            "bin": i + 1,
            "range": f"[{left:.1f}, {right:.1f}{']' if i == n_bins - 1 else ')'}",
            "count": count,
            "avg_predicted_probability": round(avg_prob, 4),
            "actual_denial_rate": round(actual_rate, 4),
            "absolute_gap": round(gap, 4),
        })

    return {
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "expected_calibration_error": round(float(ece), 4),
        "bucket_count": n_bins,
        "buckets": rows,
        "interpretation": "Lower Brier/ECE is better. Buckets should have actual_denial_rate close to avg_predicted_probability.",
    }


class ModelTrainer:
    """Full training pipeline: Gold features → trained models → governance artifacts."""

    def __init__(self, gold_dir: Path, models_dir: Path) -> None:
        self.gold_dir = Path(gold_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _load_features(self) -> pd.DataFrame:
        path = self.gold_dir / "gold_claim_features.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Gold feature table not found: {path}. Run run_gold.py first.")
        df = pd.read_parquet(path)
        logger.info("gold_features_loaded", rows=len(df), cols=len(df.columns))
        return df

    def _prepare_data(
        self,
        df: pd.DataFrame,
        features: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        missing_cols = [f for f in features if f not in df.columns]
        if missing_cols:
            raise ValueError(f"Features missing from Gold table: {missing_cols}. Re-run run_gold.py.")
        if _TARGET not in df.columns:
            raise ValueError("Gold table must include denial_flag target column.")

        X = df[features].copy()
        y = df[_TARGET].astype(int)

        bool_cols = X.select_dtypes(include="bool").columns
        X[bool_cols] = X[bool_cols].astype(int)

        # 1) Hold out test set once. Never tune threshold on this set.
        X_dev, X_test, y_dev, y_test = train_test_split(
            X, y, test_size=_TEST_SIZE, random_state=_SEED, stratify=y
        )
        # 2) Split dev into train + validation. Threshold is tuned on validation.
        X_train, X_val, y_train, y_val = train_test_split(
            X_dev,
            y_dev,
            test_size=_VALIDATION_SIZE_OF_REMAINING,
            random_state=_SEED,
            stratify=y_dev,
        )

        logger.info(
            "data_split",
            train_rows=len(X_train),
            validation_rows=len(X_val),
            test_rows=len(X_test),
            train_denied=int(y_train.sum()),
            validation_denied=int(y_val.sum()),
            test_denied=int(y_test.sum()),
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def _train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=500,
                random_state=_SEED,
                class_weight="balanced",
                C=1.0,
            )),
        ])
        pipeline.fit(X_train, y_train)
        logger.info("logistic_regression_trained")
        return pipeline

    def _train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        pos_weight = n_neg / max(n_pos, 1)

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", xgb.XGBClassifier(
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
        logger.info("xgboost_trained", n_estimators=200, scale_pos_weight=round(pos_weight, 3))
        return pipeline

    def _compute_shap_importances(
        self,
        model_pipeline: Pipeline,
        X_test: pd.DataFrame,
        feature_names: list[str],
        n_samples: int = 200,
    ) -> dict[str, float]:
        import shap

        X_imputed = model_pipeline.named_steps["imputer"].transform(X_test)
        X_sample = X_imputed[:n_samples]
        xgb_model = model_pipeline.named_steps["model"]
        explainer = shap.TreeExplainer(xgb_model)
        shap_vals = explainer.shap_values(X_sample)

        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        importance_map = {name: round(float(val), 6) for name, val in zip(feature_names, mean_abs_shap)}
        return dict(sorted(importance_map.items(), key=lambda x: -x[1]))

    def _save_json(self, path: Path, payload: dict | list) -> None:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def run(self) -> dict:
        report: dict = {}
        try:
            df = self._load_features()
            features = _load_feature_list(self.gold_dir / "feature_manifest.json")
            X_train, X_val, X_test, y_train, y_val, y_test = self._prepare_data(df, features)

            lr_pipeline = self._train_logistic_regression(X_train, y_train)
            lr_val_prob = lr_pipeline.predict_proba(X_val)[:, 1]
            lr_test_prob = lr_pipeline.predict_proba(X_test)[:, 1]
            lr_validation_default = _compute_metrics(y_val.values, lr_val_prob, threshold=0.50)
            lr_test_default = _compute_metrics(y_test.values, lr_test_prob, threshold=0.50)

            xgb_pipeline = self._train_xgboost(X_train, y_train)
            xgb_val_prob = xgb_pipeline.predict_proba(X_val)[:, 1]
            xgb_test_prob = xgb_pipeline.predict_proba(X_test)[:, 1]
            xgb_validation_default = _compute_metrics(y_val.values, xgb_val_prob, threshold=0.50)
            xgb_test_default = _compute_metrics(y_test.values, xgb_test_prob, threshold=0.50)

            # Choose recommended model by validation ROC-AUC, then validation recall.
            recommended_model = (
                "xgboost"
                if (xgb_validation_default["roc_auc"], xgb_validation_default["recall"]) >=
                   (lr_validation_default["roc_auc"], lr_validation_default["recall"])
                else "logistic_regression"
            )
            if recommended_model == "xgboost":
                recommended_pipeline = xgb_pipeline
                recommended_val_prob = xgb_val_prob
                recommended_test_prob = xgb_test_prob
            else:
                recommended_pipeline = lr_pipeline
                recommended_val_prob = lr_val_prob
                recommended_test_prob = lr_test_prob

            selected_threshold, threshold_rows = _tune_threshold(
                y_val.values,
                recommended_val_prob,
                min_recall=_MIN_DENIAL_RECALL,
            )
            tuned_threshold = float(selected_threshold["threshold"])
            risk_policy = _risk_band_policy(tuned_threshold)

            recommended_test_tuned = _compute_metrics(y_test.values, recommended_test_prob, tuned_threshold)
            calibration = _calibration_report(y_test.values, recommended_test_prob, n_bins=10)

            shap_importance = self._compute_shap_importances(xgb_pipeline, X_test, features)

            lr_path = self.models_dir / "lr_model.pkl"
            xgb_path = self.models_dir / "xgb_model.pkl"
            with open(lr_path, "wb") as f:
                pickle.dump({"pipeline": lr_pipeline, "features": features}, f)
            with open(xgb_path, "wb") as f:
                pickle.dump({"pipeline": xgb_pipeline, "features": features}, f)

            threshold_report_path = self.models_dir / "threshold_report.json"
            calibration_report_path = self.models_dir / "calibration_report.json"
            training_report_path = self.models_dir / "training_report.json"
            model_card_path = self.models_dir / "model_card.json"

            threshold_report = {
                "recommended_model": recommended_model,
                "minimum_target_recall": _MIN_DENIAL_RECALL,
                "selected_threshold": selected_threshold,
                "risk_band_policy": risk_policy,
                "validation_curve": threshold_rows,
            }
            self._save_json(threshold_report_path, threshold_report)
            self._save_json(calibration_report_path, calibration)

            training_report = {
                "features": features,
                "feature_count": len(features),
                "target": _TARGET,
                "split_policy": {
                    "train_rows": len(X_train),
                    "validation_rows": len(X_val),
                    "test_rows": len(X_test),
                    "seed": _SEED,
                },
                "class_balance": {
                    "train_denied": int(y_train.sum()),
                    "validation_denied": int(y_val.sum()),
                    "test_denied": int(y_test.sum()),
                    "test_approved": int((y_test == 0).sum()),
                },
                "logistic_regression": {
                    "validation_default_threshold": lr_validation_default,
                    "test_default_threshold": lr_test_default,
                },
                "xgboost": {
                    "validation_default_threshold": xgb_validation_default,
                    "test_default_threshold": xgb_test_default,
                },
                "recommended_model": recommended_model,
                "selected_threshold": selected_threshold,
                "risk_band_policy": risk_policy,
                "recommended_model_test_at_tuned_threshold": recommended_test_tuned,
                "calibration": calibration,
                "shap_importance": shap_importance,
            }
            self._save_json(training_report_path, training_report)

            model_card = {
                "model_name": "claim_denial_risk_model",
                "recommended_model": recommended_model,
                "problem_type": "binary classification: denial_flag 1=denied, 0=approved",
                "primary_objective": "maximize denial recall while controlling analyst review load",
                "business_threshold_policy": risk_policy,
                "selected_threshold_source": "validation set threshold tuning, test set used only for final reporting",
                "calibration_summary": {
                    "brier_score": calibration["brier_score"],
                    "expected_calibration_error": calibration["expected_calibration_error"],
                },
                "artifacts": {
                    "lr_model": str(lr_path),
                    "xgb_model": str(xgb_path),
                    "training_report": str(training_report_path),
                    "threshold_report": str(threshold_report_path),
                    "calibration_report": str(calibration_report_path),
                },
                "cloud_migration_note": (
                    "These JSON artifacts can be logged with MLflow and registered in Databricks/Unity Catalog. "
                    "Feature computation is kept deterministic so it can later move to Databricks Feature Store."
                ),
            }
            self._save_json(model_card_path, model_card)

            report.update({
                "status": "success",
                "recommended": recommended_model,
                "selected_threshold": tuned_threshold,
                "risk_band_policy": risk_policy,
                "test_recall_at_tuned_threshold": recommended_test_tuned["recall"],
                "test_precision_at_tuned_threshold": recommended_test_tuned["precision"],
                "test_f1_at_tuned_threshold": recommended_test_tuned["f1"],
                "test_roc_auc": recommended_test_tuned["roc_auc"],
                "brier_score": calibration["brier_score"],
                "expected_calibration_error": calibration["expected_calibration_error"],
                "lr_path": str(lr_path),
                "xgb_path": str(xgb_path),
                "report_path": str(training_report_path),
                "threshold_report_path": str(threshold_report_path),
                "calibration_report_path": str(calibration_report_path),
                "model_card_path": str(model_card_path),
            })
            logger.info("training_complete", **report)

        except Exception as exc:
            logger.exception("training_failed", error=str(exc))
            report["status"] = "failed"
            report["error"] = str(exc)

        return report
