"""
Claim Predictor
================
Loads a trained model pipeline and predicts denial risk for a single claim
or a batch. Returns a structured prediction dict ready for the Decision Engine
(Week 7) and FastAPI response.

Usage
-----
    predictor = ClaimPredictor.load(models_dir=Path("models"))
    result = predictor.predict(claim_features)

The claim_features dict must contain exactly the features the model was
trained on — see models/training_report.json for the full list.
Missing features are filled with the median from training (via the imputer
already fitted inside the pipeline).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# Risk level thresholds — calibrated to the synthetic label distribution
_RISK_THRESHOLDS = {
    "HIGH":   0.65,   # >= 0.65 → HIGH
    "MEDIUM": 0.40,   # 0.40–0.64 → MEDIUM
    # below 0.40 → LOW
}


def _risk_level(prob: float) -> str:
    """Convert probability to LOW / MEDIUM / HIGH label."""
    if prob >= _RISK_THRESHOLDS["HIGH"]:
        return "HIGH"
    elif prob >= _RISK_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


class ClaimPredictor:
    """
    Wraps a trained sklearn/XGBoost pipeline for inference.

    Parameters
    ----------
    pipeline  : Fitted sklearn Pipeline (from train.py).
    features  : Ordered list of feature names the pipeline expects.
    model_name: "xgboost" or "logistic_regression" — for logging/audit.
    """

    def __init__(self, pipeline: Any, features: list[str], model_name: str) -> None:
        self.pipeline   = pipeline
        self.features   = features
        self.model_name = model_name

    @classmethod
    def load(
        cls,
        models_dir: Path,
        model_name: str = "xgboost",
    ) -> "ClaimPredictor":
        """
        Load a saved model from disk.

        Parameters
        ----------
        models_dir : Directory containing lr_model.pkl / xgb_model.pkl.
        model_name : "xgboost" (default) or "logistic_regression".
                     Use training_report.json['recommended_model'] for best choice.

        Raises FileNotFoundError if the model file doesn't exist.
        """
        file_map = {
            "xgboost":              "xgb_model.pkl",
            "logistic_regression":  "lr_model.pkl",
        }
        filename = file_map.get(model_name)
        if not filename:
            raise ValueError(
                f"Unknown model_name '{model_name}'. "
                f"Choose from: {list(file_map.keys())}"
            )

        path = Path(models_dir) / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}. Run run_train.py first."
            )

        with open(path, "rb") as f:
            saved = pickle.load(f)

        logger.info("model_loaded", model=model_name, path=str(path))
        return cls(
            pipeline=saved["pipeline"],
            features=saved["features"],
            model_name=model_name,
        )

    def _build_feature_row(self, claim_features: dict) -> pd.DataFrame:
        """
        Convert a claim dict to a single-row DataFrame with the expected feature columns.
        Missing features are filled with NaN (the imputer inside the pipeline handles them).
        """
        row = {feat: claim_features.get(feat, np.nan) for feat in self.features}
        df  = pd.DataFrame([row])

        # Cast bool/int features correctly
        for col in df.columns:
            val = df[col].iloc[0]
            if isinstance(val, bool):
                df[col] = int(val)
        return df

    def predict(self, claim_features: dict) -> dict:
        """
        Predict denial risk for a single claim.

        Parameters
        ----------
        claim_features : Dict of feature_name → value.
                         Keys must match self.features.
                         Missing keys are treated as unknown (imputed).

        Returns
        -------
        {
            "claim_id":    str or None,
            "risk_score":  float (0.0–1.0, probability of denial),
            "risk_level":  "LOW" | "MEDIUM" | "HIGH",
            "model_used":  str,
            "features_received": int,
            "features_expected": int,
        }
        """
        claim_id = claim_features.get("claim_id")
        X = self._build_feature_row(claim_features)

        prob  = float(self.pipeline.predict_proba(X)[0, 1])
        level = _risk_level(prob)

        result = {
            "claim_id":           claim_id,
            "risk_score":         round(prob, 4),
            "risk_level":         level,
            "model_used":         self.model_name,
            "features_received":  sum(1 for k in claim_features if k in self.features),
            "features_expected":  len(self.features),
        }
        logger.info(
            "claim_predicted",
            claim_id=claim_id,
            risk_score=result["risk_score"],
            risk_level=level,
            model=self.model_name,
        )
        return result

    def predict_batch(self, claims: list[dict]) -> list[dict]:
        """
        Predict denial risk for a list of claim dicts.
        More efficient than calling predict() in a loop for large batches.
        """
        if not claims:
            return []

        rows    = [self._build_feature_row(c) for c in claims]
        X_batch = pd.concat(rows, ignore_index=True)

        probs  = self.pipeline.predict_proba(X_batch)[:, 1]
        results = []
        for claim, prob in zip(claims, probs):
            prob  = float(prob)
            results.append({
                "claim_id":   claim.get("claim_id"),
                "risk_score": round(prob, 4),
                "risk_level": _risk_level(prob),
                "model_used": self.model_name,
            })

        logger.info("batch_predicted", count=len(results), model=self.model_name)
        return results

    @classmethod
    def recommended(cls, models_dir: Path) -> "ClaimPredictor":
        """
        Load whichever model the training report recommended.
        Falls back to xgboost if the report is missing.
        """
        report_path = Path(models_dir) / "training_report.json"
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
            name = report.get("recommended_model", "xgboost")
        else:
            name = "xgboost"
            logger.warning("training_report_not_found_defaulting", model=name)

        return cls.load(models_dir=models_dir, model_name=name)
