#!/usr/bin/env python3
"""
Week 4 — Model Training Entry Point
====================================
Gold features → candidate models → threshold tuning → calibration report.

Usage:
    python run_train.py

Requires Gold layer (run_gold.py) to exist first.
"""

import json
import sys
from pathlib import Path

from src.config import setup_logging
from src.ml.train import ModelTrainer

BASE_DIR = Path(__file__).parent
GOLD_DIR = BASE_DIR / "data" / "gold"
MODELS_DIR = BASE_DIR / "models"


def _bar(value: float, width: int = 20) -> str:
    value = max(0.0, min(1.0, value))
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)


def main() -> int:
    setup_logging(level="INFO")

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Claim Denial Prevention — Week 4: Model Training          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    trainer = ModelTrainer(gold_dir=GOLD_DIR, models_dir=MODELS_DIR)
    report = trainer.run()

    if report["status"] != "success":
        print(f"\n  ERROR: {report.get('error')}")
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║   ✗  Training failed.                                      ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        return 1

    with open(MODELS_DIR / "training_report.json") as f:
        full = json.load(f)

    tuned = full["recommended_model_test_at_tuned_threshold"]
    risk_policy = full["risk_band_policy"]
    cal = full["calibration"]

    print("\n  Tuned decision policy:")
    print(f"    Recommended model:     {full['recommended_model'].upper()}")
    print(f"    Review threshold:      {risk_policy['medium_lower_inclusive']:.4f}")
    print(f"    Denial threshold:      {risk_policy['classification_threshold']:.4f}")
    print("    LOW < review threshold ≤ MEDIUM < denial threshold ≤ HIGH")

    print("\n  Final test-set metrics at tuned threshold:")
    print(f"    ROC-AUC:     {tuned['roc_auc']:.4f}")
    print(f"    Recall:      {tuned['recall']:.4f}  {_bar(tuned['recall'])}")
    print(f"    Precision:   {tuned['precision']:.4f}  {_bar(tuned['precision'])}")
    print(f"    F1:          {tuned['f1']:.4f}")
    print(f"    Accuracy:    {tuned['accuracy']:.4f}")
    print(f"    False neg.:  {tuned['false_negatives']}")
    print(f"    False pos.:  {tuned['false_positives']}")

    print("\n  Calibration:")
    print(f"    Brier score: {cal['brier_score']:.4f}  (lower is better)")
    print(f"    ECE:         {cal['expected_calibration_error']:.4f}  (lower is better)")

    print("\n  SHAP Feature Importances (XGBoost, top 8):")
    shap = full.get("shap_importance", {})
    if shap:
        max_imp = max(shap.values()) or 1
        for i, (feat, importance) in enumerate(list(shap.items())[:8], 1):
            print(f"    {i:>2}. {feat:<34} {importance:.4f}  {_bar(importance / max_imp)}")

    print("\n  Saved artifacts:")
    for label, path in [
        ("Models", "models/lr_model.pkl + models/xgb_model.pkl"),
        ("Training report", "models/training_report.json"),
        ("Threshold report", "models/threshold_report.json"),
        ("Calibration report", "models/calibration_report.json"),
        ("Model card", "models/model_card.json"),
    ]:
        print(f"    {label:<20} {path}")

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   ✓  Training complete. Ready for custom claim inference.   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
