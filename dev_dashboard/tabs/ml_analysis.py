"""
Dashboard Tab 3 — ML Model Analysis
======================================
Four sections:
  1. Model Performance  — metrics cards, ROC curve, confusion matrix
  2. Feature Intelligence — SHAP importance, risk score distribution
  3. Prediction Explorer — pick a real claim from Gold data, see live prediction
  4. Custom Claim Builder — toggle flags/sliders, see risk update in real time

All heavy objects (model, Gold data) are cached. The interactive sections
call predict() and explain() on demand — no page reload needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.ml.predict import ClaimPredictor
from src.ml.explain import SHAPExplainer, FEATURE_LABELS, FIX_SUGGESTIONS
from src.constants import DASHBOARD_CACHE_TTL

# ── Constants ─────────────────────────────────────────────────────────────────
_RISK_COLORS = {"HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#10B981"}
_SPECIALTY_MAP  = {"Neurology": 1, "Cardiology": 2, "Orthopedic": 3, "General": 4}
_SEVERITY_MAP   = {"Missing (code unknown)": 0, "Low severity": 1, "High severity": 2}
_SPECIALTY_INV  = {v: k for k, v in _SPECIALTY_MAP.items()}
_SEVERITY_INV   = {v: k for k, v in _SEVERITY_MAP.items()}


# ── Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_predictor(models_dir: Path) -> ClaimPredictor | None:
    try:
        return ClaimPredictor.recommended(models_dir=models_dir)
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def _load_explainer(models_dir: Path) -> SHAPExplainer | None:
    try:
        xgb_path = models_dir / "xgb_model.pkl"
        return SHAPExplainer.from_model_file(xgb_path)
    except FileNotFoundError:
        return None


@st.cache_data(ttl=DASHBOARD_CACHE_TTL, show_spinner=False)
def _load_gold(gold_dir: Path) -> pd.DataFrame:
    path = gold_dir / "gold_claim_features.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data(ttl=DASHBOARD_CACHE_TTL, show_spinner=False)
def _load_report(models_dir: Path) -> dict:
    path = models_dir / "training_report.json"
    return json.loads(path.read_text()) if path.exists() else {}


# ── ROC curve (computed on Gold features) ────────────────────────────────────

@st.cache_data(ttl=DASHBOARD_CACHE_TTL, show_spinner=False)
def _compute_roc(gold_dir: Path, models_dir: Path) -> tuple:
    """Returns (fpr, tpr, auc_lr, auc_xgb) for both models."""
    import pickle
    from sklearn.metrics import roc_curve, auc

    df   = _load_gold(gold_dir)
    report = _load_report(models_dir)
    if df.empty or not report:
        return None, None, None, None

    features = report["features"]
    X = df[features].copy()
    X[X.select_dtypes(include="bool").columns] = \
        X.select_dtypes(include="bool").astype(int)
    y = df["denial_flag"].values

    results = {}
    for name, fname in [("xgboost", "xgb_model.pkl"), ("logistic_regression", "lr_model.pkl")]:
        pkl = models_dir / fname
        if pkl.exists():
            with open(pkl, "rb") as f:
                saved = pickle.load(f)
            probs = saved["pipeline"].predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, probs)
            results[name] = (fpr, tpr, round(auc(fpr, tpr), 4))

    return results


@st.cache_data(ttl=DASHBOARD_CACHE_TTL, show_spinner=False)
def _compute_confusion(gold_dir: Path, models_dir: Path) -> dict:
    """Returns confusion matrix values and prediction counts for both models."""
    import pickle
    from sklearn.metrics import confusion_matrix

    df     = _load_gold(gold_dir)
    report = _load_report(models_dir)
    if df.empty or not report:
        return {}

    features = report["features"]
    X = df[features].copy()
    X[X.select_dtypes(include="bool").columns] = \
        X.select_dtypes(include="bool").astype(int)
    y = df["denial_flag"].values

    out = {}
    for name, fname in [("xgboost", "xgb_model.pkl"), ("logistic_regression", "lr_model.pkl")]:
        pkl = models_dir / fname
        if pkl.exists():
            with open(pkl, "rb") as f:
                saved = pickle.load(f)
            preds = saved["pipeline"].predict(X)
            cm    = confusion_matrix(y, preds)
            out[name] = {
                "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
            }
    return out


# ── Chart helpers ──────────────────────────────────────────────────────────────

def _roc_chart(roc_results: dict) -> go.Figure:
    fig = go.Figure()
    colors = {"xgboost": "#4F46E5", "logistic_regression": "#10B981"}
    labels = {"xgboost": "XGBoost", "logistic_regression": "Logistic Regression"}

    for name, (fpr, tpr, auc_val) in roc_results.items():
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f"{labels[name]} (AUC={auc_val})",
            line={"color": colors[name], "width": 2.5},
            mode="lines",
        ))

    # Diagonal reference line (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name="Random (AUC=0.50)",
        line={"color": "#9CA3AF", "dash": "dash", "width": 1},
        mode="lines",
    ))
    fig.update_layout(
        title="ROC Curve — both models on full dataset",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        height=380,
        legend={"orientation": "h", "y": -0.25},
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1.02]},
    )
    return fig


def _confusion_chart(cm: dict, model_name: str) -> go.Figure:
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    total = tn + fp + fn + tp

    z      = [[tn, fp], [fn, tp]]
    labels = [["True Negative", "False Positive"], ["False Negative", "True Positive"]]
    text   = [
        [f"<b>{tn}</b><br>({tn/total*100:.1f}%)<br>Correctly approved",
         f"<b>{fp}</b><br>({fp/total*100:.1f}%)<br>False alarm"],
        [f"<b>{fn}</b><br>({fn/total*100:.1f}%)<br>Missed denial ⚠",
         f"<b>{tp}</b><br>({tp/total*100:.1f}%)<br>Correctly flagged"],
    ]
    colorscale = [[0, "#EFF6FF"], [0.5, "#93C5FD"], [1, "#1D4ED8"]]

    fig = go.Figure(go.Heatmap(
        z=z,
        text=text,
        texttemplate="%{text}",
        colorscale=colorscale,
        showscale=False,
    ))
    fig.update_layout(
        title=f"Confusion Matrix — {model_name.replace('_', ' ').title()} (full dataset, n={total})",
        xaxis={"title": "Predicted", "tickvals": [0, 1], "ticktext": ["Approved", "Denied"]},
        yaxis={"title": "Actual",    "tickvals": [0, 1], "ticktext": ["Approved", "Denied"]},
        template="plotly_white",
        height=320,
    )
    return fig


def _shap_chart(shap_importance: dict) -> go.Figure:
    items  = list(shap_importance.items())
    labels = [FEATURE_LABELS.get(k, k) for k, _ in items]
    values = [v for _, v in items]
    max_v  = max(values) if values else 1

    colors = ["#EF4444" if v > max_v * 0.5 else "#F59E0B" if v > max_v * 0.2 else "#6B7280"
              for v in values]

    fig = go.Figure(go.Bar(
        x=values[::-1],
        y=labels[::-1],
        orientation="h",
        marker_color=colors[::-1],
        text=[f"{v:.4f}" for v in values[::-1]],
        textposition="outside",
    ))
    fig.update_layout(
        title="SHAP Feature Importance — mean |SHAP value| across test set",
        xaxis_title="Mean |SHAP value| (contribution to prediction)",
        template="plotly_white",
        height=420,
        margin={"l": 220, "r": 80, "t": 50, "b": 40},
        showlegend=False,
    )
    return fig


def _risk_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Stacked bar: Denied vs Approved broken down by risk level."""
    if "denial_risk_score" not in df.columns:
        return go.Figure()

    probs = df["denial_risk_score"].values
    df2 = df.copy()
    df2["risk_level"] = pd.cut(
        probs,
        bins=[-0.001, 0.40, 0.65, 1.001],
        labels=["LOW", "MEDIUM", "HIGH"],
    ).astype(str)

    counts = df2.groupby(["risk_level", "denial_flag"]).size().unstack(fill_value=0)

    fig = go.Figure()
    for flag, label, color in [(0, "Approved", "#10B981"), (1, "Denied", "#EF4444")]:
        if flag in counts.columns:
            fig.add_trace(go.Bar(
                name=label,
                x=["LOW", "MEDIUM", "HIGH"],
                y=[counts.loc[lvl, flag] if lvl in counts.index else 0
                   for lvl in ["LOW", "MEDIUM", "HIGH"]],
                marker_color=color,
            ))

    fig.update_layout(
        title="Risk Level Distribution — Denied vs Approved breakdown",
        xaxis_title="Risk Level",
        yaxis_title="Number of Claims",
        barmode="stack",
        template="plotly_white",
        height=320,
        legend={"orientation": "h", "y": -0.25},
    )
    return fig


def _score_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogram of denial_risk_score coloured by actual label."""
    if "denial_risk_score" not in df.columns:
        return go.Figure()

    fig = go.Figure()
    for flag, label, color in [(0, "Approved", "#10B981"), (1, "Denied", "#EF4444")]:
        subset = df[df["denial_flag"] == flag]["denial_risk_score"]
        fig.add_trace(go.Histogram(
            x=subset,
            name=label,
            marker_color=color,
            opacity=0.7,
            nbinsx=30,
        ))

    # Decision boundary
    fig.add_vline(x=0.5, line_dash="dash", line_color="#1F2937",
                  annotation_text="threshold 0.50", annotation_position="top right")

    fig.update_layout(
        title="Denial Risk Score Distribution",
        xaxis_title="Denial Risk Score",
        yaxis_title="Claims",
        barmode="overlay",
        template="plotly_white",
        height=300,
        legend={"orientation": "h", "y": -0.25},
    )
    return fig


# ── Prediction result renderer ─────────────────────────────────────────────────

def _render_prediction_result(claim_features: dict, predictor: ClaimPredictor,
                               explainer: SHAPExplainer | None) -> None:
    result = predictor.predict(claim_features)
    color  = _RISK_COLORS[result["risk_level"]]

    # Risk score card
    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("Risk Score",  f"{result['risk_score']:.0%}")
    c2.metric("Risk Level",  result["risk_level"])
    c3.metric("Model Used",  result["model_used"].replace("_", " ").title())

    # Colour-coded risk banner
    st.markdown(
        f"""<div style="background:{color}22; border-left:4px solid {color};
        padding:10px 16px; border-radius:4px; margin:8px 0;">
        <b style="color:{color}; font-size:16px;">
        {'⛔ HIGH DENIAL RISK' if result['risk_level']=='HIGH'
         else '⚠️ MEDIUM DENIAL RISK' if result['risk_level']=='MEDIUM'
         else '✅ LOW DENIAL RISK'}</b><br>
        <span style="color:#374151;">Score: {result['risk_score']:.4f}
        (threshold: 0.50 for denial)</span></div>""",
        unsafe_allow_html=True,
    )

    # SHAP explanation
    if explainer:
        explanation = explainer.explain(claim_features, top_n=3)
        st.markdown("**Top 3 Denial Reasons (SHAP)**")
        for r in explanation["top_reasons"]:
            icon  = "🔴" if r["direction"] == "increases_risk" else "🟢"
            arrow = "↑ increases risk" if r["direction"] == "increases_risk" else "↓ decreases risk"
            with st.expander(f"{icon}  {r['rank']}. {r['label']}  —  SHAP {r['shap_value']:+.4f}  ({arrow})"):
                st.write(f"**Feature:** `{r['feature']}`")
                st.write(f"**How to fix:** {r['fix']}")
    else:
        st.info("SHAP explainer not available — run `python run_train.py` to generate the XGBoost model.")


# ── Main render ────────────────────────────────────────────────────────────────

def render_ml_tab(gold_dir: Path, models_dir: Path) -> None:
    st.header("ML Model — Performance, Explainability & Predictions")

    # Guard: models not trained yet
    if not (models_dir / "training_report.json").exists():
        st.warning("Models not found. Train them first:", icon="⚠️")
        st.code("python run_gold.py\npython run_train.py", language="bash")
        return

    with st.spinner("Loading model and data..."):
        report    = _load_report(models_dir)
        df        = _load_gold(gold_dir)
        predictor = _load_predictor(models_dir)
        explainer = _load_explainer(models_dir)
        roc_data  = _compute_roc(gold_dir, models_dir)
        cm_data   = _compute_confusion(gold_dir, models_dir)

    if predictor is None:
        st.error("Could not load model. Run `python run_train.py`.")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Model Performance
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("1 — Model Performance")

    # Key metrics row
    xgb = report.get("xgboost", {})
    lr  = report.get("logistic_regression", {})
    rec = report.get("recommended_model", "xgboost")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("XGB ROC-AUC",   f"{xgb.get('roc_auc', 0):.4f}", "recommended ✓" if rec == "xgboost" else "")
    c2.metric("XGB Recall",    f"{xgb.get('recall', 0):.4f}",  "primary metric")
    c3.metric("XGB Precision", f"{xgb.get('precision', 0):.4f}")
    c4.metric("XGB F1",        f"{xgb.get('f1', 0):.4f}")
    c5.metric("XGB Accuracy",  f"{xgb.get('accuracy', 0):.4f}")

    st.caption(
        "**Recall is the primary metric** — a missed denial (false negative) costs more "
        "than a false alarm in a pre-submission validation system."
    )

    # ROC + Confusion matrix side by side
    c_left, c_right = st.columns(2)
    with c_left:
        if roc_data:
            st.plotly_chart(_roc_chart(roc_data), key="ml_roc", width="stretch")

    with c_right:
        model_choice = st.selectbox(
            "Confusion matrix for:",
            ["xgboost", "logistic_regression"],
            format_func=lambda x: x.replace("_", " ").title(),
            key="cm_model_select",
        )
        if cm_data and model_choice in cm_data:
            st.plotly_chart(_confusion_chart(cm_data[model_choice], model_choice),
                            key="ml_cm", width="stretch")
            cm = cm_data[model_choice]
            st.caption(
                f"**Missed denials (FN): {cm['fn']}** — claims that should be flagged but weren't. "
                f"**False alarms (FP): {cm['fp']}** — clean claims incorrectly flagged."
            )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Feature Intelligence
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("2 — Feature Intelligence")
    st.caption("What the model actually learned — which features drive denial predictions.")

    c_left, c_right = st.columns([3, 2])
    with c_left:
        if report.get("shap_importance"):
            st.plotly_chart(_shap_chart(report["shap_importance"]),
                            key="ml_shap", width="stretch")
    with c_right:
        st.markdown("**Key insight from SHAP**")
        st.markdown(
            "- `diagnosis_code_missing` dominates — its absence is the single strongest denial predictor\n"
            "- `proc_no_diag` is second — billing a procedure without clinical justification\n"
            "- `severity_encoded` is third — the model discovered High-severity diagnoses with missing procedures are disproportionately risky. **The label didn't encode severity directly — this is genuine ML signal**\n"
            "- Provider-level features (`violation_rate`, `claim_count`) contribute, confirming provider history is a useful signal"
        )

    c_left, c_right = st.columns(2)
    with c_left:
        if not df.empty:
            st.plotly_chart(_score_histogram(df), key="ml_hist", width="stretch")
    with c_right:
        if not df.empty:
            st.plotly_chart(_risk_distribution_chart(df), key="ml_dist", width="stretch")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Prediction Explorer (real claims from Gold data)
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("3 — Prediction Explorer")
    st.caption(
        "Pick any real claim from the Gold dataset. The model predicts its denial risk "
        "and explains the top 3 reasons. This confirms the model works on actual data."
    )

    if df.empty:
        st.warning("Gold data not found. Run `python run_gold.py`.")
    else:
        features = report.get("features", [])

        # Dropdown with denial flag shown so you can pick known-denied claims
        claim_options = {}
        for _, row in df.iterrows():
            label = (
                f"{row['claim_id']}  |  "
                f"{'⛔ Denied' if row['denial_flag'] == 1 else '✅ Approved'}  |  "
                f"diag={'✗' if row.get('diagnosis_code_missing') else '✓'}  "
                f"proc={'✗' if row.get('procedure_code_missing') else '✓'}  "
                f"proc_no_diag={'⚠' if row.get('proc_no_diag') else '–'}"
            )
            claim_options[label] = row.to_dict()

        selected_label = st.selectbox(
            "Select a claim:",
            list(claim_options.keys()),
            key="claim_explorer_select",
        )
        selected_claim = claim_options[selected_label]

        # Show raw feature values
        with st.expander("Show claim feature values"):
            display = {
                FEATURE_LABELS.get(k, k): str(v)  # str() prevents Arrow bool/int type clash
                for k, v in selected_claim.items()
                if k in features
            }
            feat_df = pd.DataFrame(list(display.items()), columns=["Feature", "Value"])
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

        # Actual label vs prediction
        actual = "⛔ Denied" if selected_claim.get("denial_flag") == 1 else "✅ Approved"
        st.markdown(f"**Actual label (synthetic):** {actual}")
        st.markdown("**Model prediction:**")

        _render_prediction_result(selected_claim, predictor, explainer)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Custom Claim Builder
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("4 — Custom Claim Builder")
    st.caption(
        "Build any hypothetical claim by toggling flags and adjusting values. "
        "See how each change affects the risk score instantly."
    )

    with st.form("custom_claim_form"):
        st.markdown("**Structural Flags** — what's missing from the claim")
        fc1, fc2, fc3 = st.columns(3)
        diag_missing  = fc1.checkbox("diagnosis_code missing",  value=False, key="cb_diag")
        proc_missing  = fc2.checkbox("procedure_code missing",  value=False, key="cb_proc")
        amt_missing   = fc3.checkbox("billed_amount missing",   value=False, key="cb_amt")

        st.markdown("**Business Logic Flags**")
        fb1, fb2 = st.columns(2)
        proc_no_diag = fb1.checkbox("proc_no_diag  (procedure without diagnosis)",
                                     value=False, key="cb_pnd")
        diag_no_proc = fb2.checkbox("diag_no_proc  (diagnosis without procedure)",
                                     value=False, key="cb_dnp")

        st.markdown("**Numeric Features**")
        fn1, fn2 = st.columns(2)
        billed_raw = fn1.number_input(
            "Billed amount (₹)", min_value=0, max_value=100000,
            value=22000, step=1000, key="ni_billed",
        )
        deviation = fn2.slider(
            "Billing deviation from expected (%)",
            min_value=-100, max_value=500, value=150, step=10, key="sl_dev",
        )
        fp1, fp2, fp3 = st.columns(3)
        prov_count   = fp1.slider("Provider claim count",    34, 63, 48, key="sl_pcc")
        prov_rate    = fp2.slider("Provider violation rate", 0.95, 1.62, 1.28,
                                   step=0.05, key="sl_pvr")
        pat_count    = fp3.slider("Patient claim count",     1, 8, 2, key="sl_pc")

        st.markdown("**Categorical**")
        fc1, fc2, fc3 = st.columns(3)
        severity_label  = fc1.selectbox("Diagnosis severity",
                                         list(_SEVERITY_MAP.keys()), index=1, key="sel_sev")
        specialty_label = fc2.selectbox("Provider specialty",
                                         list(_SPECIALTY_MAP.keys()), index=0, key="sel_spec")
        is_high_cost    = fc3.checkbox("High-cost claim", value=False, key="cb_hc")

        submitted = st.form_submit_button("🔍  Predict Denial Risk", use_container_width=True)

    if submitted:
        custom_claim = {
            "claim_id":                "CUSTOM",
            "diagnosis_code_missing":  int(diag_missing),
            "procedure_code_missing":  int(proc_missing),
            "billed_amount_missing":   int(amt_missing),
            "proc_no_diag":            int(proc_no_diag),
            "diag_no_proc":            int(diag_no_proc),
            "billed_deviation_capped": float(min(deviation, 500)),
            "log_billed_amount":       float(np.log1p(billed_raw)) if billed_raw > 0 else 0.0,
            "is_high_cost":            int(is_high_cost),
            "provider_claim_count":    float(prov_count),
            "provider_violation_rate": float(prov_rate),
            "patient_claim_count":     float(pat_count),
            "severity_encoded":        float(_SEVERITY_MAP[severity_label]),
            "specialty_encoded":       float(_SPECIALTY_MAP[specialty_label]),
        }
        st.markdown("---")
        st.markdown("**Prediction for custom claim:**")
        _render_prediction_result(custom_claim, predictor, explainer)

    # Training data summary at the bottom
    with st.expander("Training details"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train rows",   report.get("train_rows", 0))
        c2.metric("Test rows",    report.get("test_rows", 0))
        c3.metric("Features",     report.get("feature_count", 0))
        c4.metric("Target",       report.get("target", "—"))
        st.caption(
            f"Recommended model: **{report.get('recommended_model', '—').upper()}** · "
            f"Train denied: {report.get('train_rows', 0) - report.get('test_rows', 0)} · "
            "Split: 70/30 stratified"
        )
