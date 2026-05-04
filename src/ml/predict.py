"""
Claim Predictor
===============
Loads a trained model and predicts denial risk for model-ready features.
It uses the tuned threshold and risk-band policy saved by run_train.py.

The predictor does not build features from raw custom claims directly. For raw
claim dictionaries, use src.inference.feature_builder.CustomClaimFeatureBuilder
or src.inference.claim_service. This separation prevents training-serving skew.
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

_DEFAULT_RISK_POLICY = {
    "low_upper_exclusive": 0.40,
    "medium_lower_inclusive": 0.40,
    "medium_upper_exclusive": 0.65,
    "high_lower_inclusive": 0.65,
    "classification_threshold": 0.65,
    "policy": "fallback policy: LOW <0.40, MEDIUM 0.40-0.65, HIGH >=0.65",
}


def _risk_level(prob: float, risk_policy: dict) -> str:
    high = float(risk_policy.get("high_lower_inclusive", risk_policy.get("classification_threshold", 0.65)))
    medium = float(risk_policy.get("medium_lower_inclusive", 0.40))
    if prob >= high:
        return "HIGH"
    if prob >= medium:
        return "MEDIUM"
    return "LOW"


class ClaimPredictor:
    """Wraps a saved sklearn/XGBoost pipeline for inference."""

    def __init__(self, pipeline: Any, features: list[str], model_name: str, risk_policy: dict | None = None) -> None:
        self.pipeline = pipeline
        self.features = features
        self.model_name = model_name
        self.risk_policy = risk_policy or _DEFAULT_RISK_POLICY

    @classmethod
    def load(cls, models_dir: Path, model_name: str = "xgboost") -> "ClaimPredictor":
        file_map = {
            "xgboost": "xgb_model.pkl",
            "logistic_regression": "lr_model.pkl",
        }
        filename = file_map.get(model_name)
        if not filename:
            raise ValueError(f"Unknown model_name '{model_name}'. Choose from: {list(file_map.keys())}")

        models_dir = Path(models_dir)
        path = models_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}. Run run_train.py first.")

        with open(path, "rb") as f:
            saved = pickle.load(f)

        risk_policy = _DEFAULT_RISK_POLICY
        report_path = models_dir / "training_report.json"
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
            risk_policy = report.get("risk_band_policy", risk_policy)

        logger.info("model_loaded", model=model_name, path=str(path))
        return cls(
            pipeline=saved["pipeline"],
            features=saved["features"],
            model_name=model_name,
            risk_policy=risk_policy,
        )

    @classmethod
    def recommended(cls, models_dir: Path) -> "ClaimPredictor":
        report_path = Path(models_dir) / "training_report.json"
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
            name = report.get("recommended_model", "xgboost")
        else:
            name = "xgboost"
            logger.warning("training_report_not_found_defaulting", model=name)
        return cls.load(models_dir=models_dir, model_name=name)

    def _build_feature_row(self, claim_features: dict) -> pd.DataFrame:
        row = {feat: claim_features.get(feat, np.nan) for feat in self.features}
        df = pd.DataFrame([row])
        bool_cols = df.select_dtypes(include="bool").columns
        df[bool_cols] = df[bool_cols].astype(int)
        return df

    def predict(self, claim_features: dict) -> dict:
        """Predict denial risk for a single model-ready feature dict."""
        claim_id = claim_features.get("claim_id")
        X = self._build_feature_row(claim_features)

        prob = float(self.pipeline.predict_proba(X)[0, 1])
        classification_threshold = float(self.risk_policy.get("classification_threshold", 0.65))
        predicted_denial = int(prob >= classification_threshold)
        level = _risk_level(prob, self.risk_policy)

        result = {
            "claim_id": claim_id,
            "risk_score": round(prob, 4),
            "risk_level": level,
            "predicted_denial": predicted_denial,
            "classification_threshold": round(classification_threshold, 4),
            "review_threshold": round(float(self.risk_policy.get("medium_lower_inclusive", 0.40)), 4),
            "model_used": self.model_name,
            "features_received": sum(1 for k in claim_features if k in self.features),
            "features_expected": len(self.features),
            "risk_policy": self.risk_policy.get("policy"),
        }
        logger.info(
            "claim_predicted",
            claim_id=claim_id,
            risk_score=result["risk_score"],
            risk_level=level,
            predicted_denial=predicted_denial,
            model=self.model_name,
        )
        return result

    def predict_batch(self, claims: list[dict]) -> list[dict]:
        if not claims:
            return []
        X_batch = pd.concat([self._build_feature_row(c) for c in claims], ignore_index=True)
        probs = self.pipeline.predict_proba(X_batch)[:, 1]
        classification_threshold = float(self.risk_policy.get("classification_threshold", 0.65))

        results = []
        for claim, prob in zip(claims, probs):
            prob = float(prob)
            results.append({
                "claim_id": claim.get("claim_id"),
                "risk_score": round(prob, 4),
                "risk_level": _risk_level(prob, self.risk_policy),
                "predicted_denial": int(prob >= classification_threshold),
                "classification_threshold": round(classification_threshold, 4),
                "model_used": self.model_name,
            })
        logger.info("batch_predicted", count=len(results), model=self.model_name)
        return results
