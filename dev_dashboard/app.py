"""
Dev Dashboard — Entry Point
==============================
Development-only Streamlit dashboard for inspecting Bronze and Silver data.

NOT production UI. Run from project root:
    streamlit run dev_dashboard/app.py

Requires both ingestion (run_ingestion.py) and optionally silver
(run_silver.py) to have been run first.
"""

import sys
from pathlib import Path

# Add project root to Python path so src/ and dev_dashboard/ are importable
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "dev_dashboard"))

import streamlit as st

from src.config import setup_logging
from tabs.raw_data import render_raw_tab
from tabs.clean_data import render_clean_tab
from tabs.ml_analysis import render_ml_tab

# Configure logging (silent for the dashboard — logs go to terminal)
setup_logging(level="WARNING")

# ── Paths ──────────────────────────────────────────────────────────────────────
BRONZE_DIR = _ROOT / "data" / "bronze"
SILVER_DIR = _ROOT / "data" / "silver"
GOLD_DIR   = _ROOT / "data" / "gold"
MODELS_DIR = _ROOT / "models"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Claim Denial Prevention — Dev Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🏥 Claim Denial Prevention — Dev Dashboard")
st.caption("Internal tool · Data exploration · Not for end users")

# ── Sidebar: pipeline controls ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Pipeline Controls")
    st.caption("Run pipelines and refresh dashboard data.")

    if st.button("🔄 Re-run Ingestion", use_container_width=True):
        import subprocess
        result = subprocess.run(
            [sys.executable, str(_ROOT / "run_ingestion.py")],
            capture_output=True, text=True, cwd=str(_ROOT),
        )
        if result.returncode == 0:
            st.success("Ingestion complete. Refresh the page.")
            st.cache_data.clear()
        else:
            st.error(f"Ingestion failed:\n{result.stderr[:500]}")

    if st.button("⚙️ Re-run Gold Pipeline", use_container_width=True):
        import subprocess
        result = subprocess.run(
            [sys.executable, str(_ROOT / "run_gold.py")],
            capture_output=True, text=True, cwd=str(_ROOT),
        )
        if result.returncode == 0:
            st.success("Gold pipeline complete. Refresh the page.")
            st.cache_data.clear()
        else:
            st.error(f"Gold failed:\n{result.stderr[:500]}")

    if st.button("🤖 Re-run Model Training", use_container_width=True):
        import subprocess
        result = subprocess.run(
            [sys.executable, str(_ROOT / "run_train.py")],
            capture_output=True, text=True, cwd=str(_ROOT),
        )
        if result.returncode == 0:
            st.success("Training complete. Refresh the page.")
            st.cache_data.clear()
            st.cache_resource.clear()
        else:
            st.error(f"Training failed:\n{result.stderr[:500]}")

    if st.button("🧹 Re-run Silver Cleaning", use_container_width=True):
        import subprocess
        result = subprocess.run(
            [sys.executable, str(_ROOT / "run_silver.py")],
            capture_output=True, text=True, cwd=str(_ROOT),
        )
        if result.returncode == 0:
            st.success("Silver pipeline complete. Refresh the page.")
            st.cache_data.clear()
        else:
            st.error(f"Silver pipeline failed:\n{result.stderr[:500]}")

    st.divider()
    st.caption("Data locations:")
    st.code(f"Bronze: data/bronze/\nSilver: data/silver/", language="text")

    # Check data status
    st.divider()
    st.markdown("**Data Status**")
    bronze_ok = (BRONZE_DIR / "claims" / "claims_bronze.parquet").exists()
    silver_ok  = (SILVER_DIR / "claims" / "claims_silver.parquet").exists()
    gold_ok   = (GOLD_DIR   / "gold_claim_features.parquet").exists()
    models_ok = (MODELS_DIR / "training_report.json").exists()
    st.markdown(f"{'✅' if bronze_ok else '❌'} Bronze layer")
    st.markdown(f"{'✅' if silver_ok  else '⚠️'} Silver layer")
    st.markdown(f"{'✅' if gold_ok   else '⚠️'} Gold layer")
    st.markdown(f"{'✅' if models_ok else '⚠️'} ML models")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_raw, tab_clean, tab_ml = st.tabs([
    "📊 Raw Data (Bronze)",
    "✨ Clean Data (Silver)",
    "🤖 ML Model (Week 4)",
])

with tab_raw:
    render_raw_tab(bronze_dir=BRONZE_DIR)

with tab_clean:
    render_clean_tab(bronze_dir=BRONZE_DIR, silver_dir=SILVER_DIR)

with tab_ml:
    render_ml_tab(gold_dir=GOLD_DIR, models_dir=MODELS_DIR)
