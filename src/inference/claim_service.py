"""
Claim Denial Service
====================
Small local service wrapper for custom claim builder workflows.

This is the local equivalent of what Week 7/FastAPI should expose later:
raw claim → deterministic feature builder → ML predictor → risk result.
The LLM/agent layer should call this service/tool, not invent a risk score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.inference.feature_builder import CustomClaimFeatureBuilder
from src.ml.predict import ClaimPredictor


class ClaimDenialService:
    def __init__(self, feature_builder: CustomClaimFeatureBuilder, predictor: ClaimPredictor) -> None:
        self.feature_builder = feature_builder
        self.predictor = predictor

    @classmethod
    def load(cls, gold_dir: Path, models_dir: Path) -> "ClaimDenialService":
        return cls(
            feature_builder=CustomClaimFeatureBuilder.load(gold_dir),
            predictor=ClaimPredictor.recommended(models_dir),
        )

    def score_claim(self, claim: dict[str, Any]) -> dict[str, Any]:
        features = self.feature_builder.build(claim)
        prediction = self.predictor.predict(features)
        return {
            "claim_id": claim.get("claim_id"),
            "prediction": prediction,
            "features": features,
            "next_action": self._next_action(prediction["risk_level"]),
        }

    @staticmethod
    def _next_action(risk_level: str) -> str:
        if risk_level == "HIGH":
            return "Block or escalate before submission; show denial reasons and fixes."
        if risk_level == "MEDIUM":
            return "Send to billing analyst review; do not treat as automatically safe."
        return "Allow normal submission after rule checks pass."
