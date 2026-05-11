"""
Dev Dashboard — Entry Point
==============================
Development-only Streamlit dashboard for inspecting pipeline outputs.

NOT production UI. Run from project root:
    streamlit run dev_dashboard/app.py
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
from tabs.policy_rag import render_policy_rag_tab
from tabs.explainability import render_explainability_tab

setup_logging(level="WARNING")

BRONZE_DIR = _ROOT / "data" / "bronze"
SILVER_DIR = _ROOT / "data" / "silver"
GOLD_DIR   = _ROOT / "data" / "gold"
MODELS_DIR = _ROOT / "models"

st.set_page_config(
    page_title="Claim Denial Prevention — Dev Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Claim Denial Prevention — Dev Dashboard")
st.caption("Internal tool · Data exploration · Not for end users")

with st.sidebar:
    st.header("Pipeline Controls")
    st.caption("Run pipelines and refresh dashboard data.")

    def _run_script(script_name: str):
        import subprocess
        return subprocess.run(
            [sys.executable, str(_ROOT / script_name)],
            capture_output=True,
            text=True,
            cwd=str(_ROOT),
        )

    if st.button("🔄 Re-run Ingestion", use_container_width=True):
        result = _run_script("run_ingestion.py")
        if result.returncode == 0:
            st.success("Ingestion complete. Refresh the page.")
            st.cache_data.clear()
        else:
            st.error(f"Ingestion failed:\n{(result.stderr or result.stdout)[:800]}")

    if st.button("🧹 Re-run Silver Cleaning", use_container_width=True):
        result = _run_script("run_silver.py")
        if result.returncode == 0:
            st.success("Silver pipeline complete. Refresh the page.")
            st.cache_data.clear()
        else:
            st.error(f"Silver pipeline failed:\n{(result.stderr or result.stdout)[:800]}")

    if st.button("⚙️ Re-run Gold Pipeline", use_container_width=True):
        result = _run_script("run_gold.py")
        if result.returncode == 0:
            st.success("Gold pipeline complete. Refresh the page.")
            st.cache_data.clear()
        else:
            st.error(f"Gold failed:\n{(result.stderr or result.stdout)[:800]}")

    if st.button("🤖 Re-run Model Training", use_container_width=True):
        result = _run_script("run_train.py")
        if result.returncode == 0:
            st.success("Training complete. Refresh the page.")
            st.cache_data.clear()
            st.cache_resource.clear()
        else:
            st.error(f"Training failed:\n{(result.stderr or result.stdout)[:800]}")

    if st.button("🧠 Re-run Explainability Explanations", use_container_width=True):
        result = _run_script("run_explain.py")
        if result.returncode == 0:
            st.success("Explanations complete. Refresh the page.")
            st.cache_data.clear()
            st.cache_resource.clear()
        else:
            st.error(f"Explainability failed:\n{(result.stderr or result.stdout)[:800]}")

    st.divider()
    st.caption("Data locations:")
    st.code(
        "Bronze: data/bronze/\n"
        "Silver: data/silver/\n"
        "Gold: data/gold/\n"
        "Models: models/",
        language="text",
    )

    st.divider()
    st.markdown("**Data Status**")
    bronze_ok = (BRONZE_DIR / "claims" / "claims_bronze.parquet").exists()
    silver_ok = (SILVER_DIR / "claims" / "claims_silver.parquet").exists()
    gold_ok = (GOLD_DIR / "gold_claim_features.parquet").exists()
    models_ok = (MODELS_DIR / "training_report.json").exists()
    explain_ok = (GOLD_DIR / "gold_claim_explanation_summary.parquet").exists()
    st.markdown(f"{'✅' if bronze_ok else '❌'} Bronze layer")
    st.markdown(f"{'✅' if silver_ok else '⚠️'} Silver layer")
    st.markdown(f"{'✅' if gold_ok else '⚠️'} Gold layer")
    st.markdown(f"{'✅' if models_ok else '⚠️'} ML models")
    st.markdown(f"{'✅' if explain_ok else '⚠️'} Explainability explanations")

tab_raw, tab_clean, tab_ml, tab_xai, tab_rag = st.tabs([
    "Raw Data",
    "Clean Data",
    "Risk Model",
    "Risk Explanations",
    "Policy Evidence",
])

with tab_raw:
    render_raw_tab(bronze_dir=BRONZE_DIR)

with tab_clean:
    render_clean_tab(bronze_dir=BRONZE_DIR, silver_dir=SILVER_DIR)

with tab_ml:
    render_ml_tab(gold_dir=GOLD_DIR, models_dir=MODELS_DIR)

with tab_xai:
    render_explainability_tab(gold_dir=GOLD_DIR, models_dir=MODELS_DIR)

with tab_rag:
    render_policy_rag_tab(root_dir=_ROOT, gold_dir=GOLD_DIR, models_dir=MODELS_DIR)
