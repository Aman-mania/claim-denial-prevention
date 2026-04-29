#!/usr/bin/env python3
"""
Week 4 — Model Training Entry Point
=====================================
Gold features → Logistic Regression + XGBoost → saved models + training report.

Usage:
    python run_train.py

Requires Gold layer (run_gold.py) to exist first.
"""

import sys
from pathlib import Path

from src.config import setup_logging
from src.ml.train import ModelTrainer

BASE_DIR   = Path(__file__).parent
GOLD_DIR   = BASE_DIR / "data" / "gold"
MODELS_DIR = BASE_DIR / "models"


def _bar(value: float, width: int = 20) -> str:
    """Simple ASCII progress bar."""
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
    report  = trainer.run()

    if report["status"] != "success":
        print(f"\n  ERROR: {report.get('error')}")
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║   ✗  Training failed.                                        ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        return 1

    print("\n  ┌──────────────────────────────────────────────────────────┐")
    print("  │  Model Evaluation Results (held-out test set)            │")
    print("  ├──────────────────────┬──────────────────┬────────────────┤")
    print(f"  │ {'Metric':<20}  │ {'Logistic Reg':>16} │ {'XGBoost':>14} │")
    print("  ├──────────────────────┼──────────────────┼────────────────┤")

    lr  = report.get("lr_roc_auc", 0)
    xgb = report.get("xgb_roc_auc", 0)

    # Load full metrics from training report for the table
    import json
    report_path = MODELS_DIR / "training_report.json"
    with open(report_path) as f:
        full = json.load(f)

    lr_m  = full["logistic_regression"]
    xgb_m = full["xgboost"]

    for label, key in [("ROC-AUC", "roc_auc"), ("Recall", "recall"),
                        ("Precision", "precision"), ("F1", "f1"), ("Accuracy", "accuracy")]:
        lv = lr_m[key]
        xv = xgb_m[key]
        winner_l = " ←" if lv > xv else "  "
        winner_x = " ←" if xv > lv else "  "
        print(f"  │ {label:<20}  │ {lv:>14.4f}{winner_l} │ {xv:>12.4f}{winner_x} │")

    print("  ├──────────────────────┼──────────────────┼────────────────┤")
    support = xgb_m["support"]
    print(f"  │ {'Test set size':<20}  │ {support['total']:>16,} │ {support['total']:>14,} │")
    print(f"  │ {'  Denied':<20}  │ {support['denied']:>16,} │ {support['denied']:>14,} │")
    print(f"  │ {'  Approved':<20}  │ {support['approved']:>16,} │ {support['approved']:>14,} │")
    print("  └──────────────────────┴──────────────────┴────────────────┘")

    print(f"\n  Recommended model: {full['recommended_model'].upper()}")

    print(f"\n  SHAP Feature Importances (XGBoost, top 8):")
    for i, (feat, importance) in enumerate(list(full["shap_importance"].items())[:8], 1):
        bar = _bar(importance / max(full["shap_importance"].values()))
        print(f"    {i:>2}. {feat:<30} {importance:.4f}  {bar}")

    print(f"\n  Models saved to: models/")
    print(f"  Training report: models/training_report.json")
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   ✓  Training complete. Ready for Week 5 RAG pipeline.      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
