"""
Dashboard Tab 3 — ML Model Analysis
====================================
Shows Week 4 model performance, explainability, prediction explorer, and a
raw-claim-based custom claim builder.

Important: the custom builder now calls ClaimDenialService. It no longer sends
manual model-feature toggles directly to the predictor, which prevents
inconsistent states such as billed_amount present + billed_amount_missing=True.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve, auc

from src.constants import DASHBOARD_CACHE_TTL, SENTINEL_MISSING
from src.inference.claim_service import ClaimDenialService
from src.ml.explain import SHAPExplainer, FEATURE_LABELS

_RISK_COLORS = {"HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#10B981"}


@st.cache_data(ttl=DASHBOARD_CACHE_TTL, show_spinner=False)
def _load_report(models_dir: Path) -> dict:
    path = models_dir / "training_report.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data(ttl=DASHBOARD_CACHE_TTL, show_spinner=False)
def _load_gold(gold_dir: Path) -> pd.DataFrame:
    path = gold_dir / "gold_claim_features.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_resource(show_spinner=False)
def _load_service(gold_dir: Path, models_dir: Path) -> ClaimDenialService | None:
    try:
        return ClaimDenialService.load(gold_dir=gold_dir, models_dir=models_dir)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _load_explainer(models_dir: Path) -> SHAPExplainer | None:
    try:
        return SHAPExplainer.from_model_file(models_dir / "xgb_model.pkl")
    except Exception:
        return None


def _risk_policy(report: dict) -> dict:
    return report.get("risk_band_policy", {
        "medium_lower_inclusive": 0.40,
        "classification_threshold": 0.65,
        "high_lower_inclusive": 0.65,
    })


def _classification_threshold(report: dict) -> float:
    return float(_risk_policy(report).get("classification_threshold", 0.65))


def _metric_block(report: dict) -> dict:
    return report.get("recommended_model_test_at_tuned_threshold", {})


def _load_saved_model(models_dir: Path, fname: str):
    with open(models_dir / fname, "rb") as f:
        return pickle.load(f)


def _prepare_X(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = df[features].copy()
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)
    return X


@st.cache_data(ttl=DASHBOARD_CACHE_TTL, show_spinner=False)
def _compute_roc(gold_dir: Path, models_dir: Path) -> dict:
    df = _load_gold(gold_dir)
    report = _load_report(models_dir)
    if df.empty or not report:
        return {}

    features = report.get("features", [])
    if not features:
        return {}
    X = _prepare_X(df, features)
    y = df["denial_flag"].values

    results = {}
    for name, fname in [("xgboost", "xgb_model.pkl"), ("logistic_regression", "lr_model.pkl")]:
        pkl = models_dir / fname
        if pkl.exists():
            saved = _load_saved_model(models_dir, fname)
            probs = saved["pipeline"].predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, probs)
            results[name] = (fpr, tpr, round(auc(fpr, tpr), 4))
    return results


@st.cache_data(ttl=DASHBOARD_CACHE_TTL, show_spinner=False)
def _compute_confusion(gold_dir: Path, models_dir: Path) -> dict:
    df = _load_gold(gold_dir)
    report = _load_report(models_dir)
    if df.empty or not report:
        return {}

    features = report.get("features", [])
    if not features:
        return {}

    threshold = _classification_threshold(report)
    X = _prepare_X(df, features)
    y = df["denial_flag"].values

    out = {}
    for name, fname in [("xgboost", "xgb_model.pkl"), ("logistic_regression", "lr_model.pkl")]:
        pkl = models_dir / fname
        if pkl.exists():
            saved = _load_saved_model(models_dir, fname)
            probs = saved["pipeline"].predict_proba(X)[:, 1]
            preds = (probs >= threshold).astype(int)
            cm = confusion_matrix(y, preds, labels=[0, 1])
            out[name] = {
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
                "threshold": threshold,
            }
    return out


def _roc_chart(roc_results: dict) -> go.Figure:
    fig = go.Figure()
    labels = {"xgboost": "XGBoost", "logistic_regression": "Logistic Regression"}

    for name, (fpr, tpr, auc_val) in roc_results.items():
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            name=f"{labels.get(name, name)} (AUC={auc_val})",
            mode="lines",
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        name="Random (AUC=0.50)",
        line={"dash": "dash"},
        mode="lines",
    ))
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        height=360,
        legend={"orientation": "h", "y": -0.25},
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1.02]},
    )
    return fig


def _confusion_chart(cm: dict, model_name: str) -> go.Figure:
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    total = max(tn + fp + fn + tp, 1)
    z = [[tn, fp], [fn, tp]]
    text = [
        [f"<b>{tn}</b><br>({tn/total*100:.1f}%)<br>Correctly approved",
         f"<b>{fp}</b><br>({fp/total*100:.1f}%)<br>Manual review false alarm"],
        [f"<b>{fn}</b><br>({fn/total*100:.1f}%)<br>Missed denial ⚠",
         f"<b>{tp}</b><br>({tp/total*100:.1f}%)<br>Correctly flagged"],
    ]

    fig = go.Figure(go.Heatmap(
        z=z,
        text=text,
        texttemplate="%{text}",
        showscale=False,
    ))
    fig.update_layout(
        title=f"Confusion Matrix — {model_name.replace('_', ' ').title()} @ threshold {cm.get('threshold', 0.5):.2f}",
        xaxis={"title": "Predicted", "tickvals": [0, 1], "ticktext": ["Approved/Low", "Denied/High"]},
        yaxis={"title": "Actual", "tickvals": [0, 1], "ticktext": ["Approved", "Denied"]},
        template="plotly_white",
        height=320,
    )
    return fig


def _shap_chart(shap_importance: dict) -> go.Figure:
    items = list(shap_importance.items())[:15]
    labels = [FEATURE_LABELS.get(k, k) for k, _ in items]
    values = [v for _, v in items]

    fig = go.Figure(go.Bar(
        x=values[::-1],
        y=labels[::-1],
        orientation="h",
        text=[f"{v:.4f}" for v in values[::-1]],
        textposition="outside",
    ))
    fig.update_layout(
        title="SHAP Feature Importance — mean |raw log-odds contribution|",
        xaxis_title="Mean |SHAP value|",
        template="plotly_white",
        height=430,
        margin={"l": 220, "r": 80, "t": 50, "b": 40},
        showlegend=False,
    )
    return fig


def _score_histogram(df: pd.DataFrame, report: dict) -> go.Figure:
    if df.empty:
        return go.Figure()

    # Prefer model probabilities only when saved as denial_risk_score for legacy data.
    score_col = "denial_risk_score"
    if score_col not in df.columns or df[score_col].isna().all():
        return go.Figure().update_layout(
            title="Risk score distribution unavailable until model probabilities are materialized",
            template="plotly_white",
            height=300,
        )

    fig = go.Figure()
    for flag, label in [(0, "Approved"), (1, "Denied")]:
        subset = df[df["denial_flag"] == flag][score_col].dropna()
        fig.add_trace(go.Histogram(x=subset, name=label, opacity=0.7, nbinsx=30))

    policy = _risk_policy(report)
    fig.add_vline(x=policy.get("medium_lower_inclusive", 0.4), line_dash="dash",
                  annotation_text="review threshold")
    fig.add_vline(x=policy.get("classification_threshold", 0.65), line_dash="dash",
                  annotation_text="denial threshold")

    fig.update_layout(
        title="Risk Score Distribution",
        xaxis_title="Risk Score",
        yaxis_title="Claims",
        barmode="overlay",
        template="plotly_white",
        height=300,
        legend={"orientation": "h", "y": -0.25},
    )
    return fig


def _render_prediction_payload(prediction: dict, features: dict, explainer: SHAPExplainer | None) -> None:
    color = _RISK_COLORS.get(prediction["risk_level"], "#6B7280")

    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    c1.metric("Risk Score", f"{prediction['risk_score']:.0%}")
    c2.metric("Risk Level", prediction["risk_level"])
    c3.metric("Predicted Denial", "Yes" if prediction.get("predicted_denial") else "No")
    c4.metric("Model Used", prediction["model_used"].replace("_", " ").title())

    st.markdown(
        f"""<div style="background:{color}22; border-left:4px solid {color};
        padding:10px 16px; border-radius:4px; margin:8px 0;">
        <b style="color:{color}; font-size:16px;">
        {'⛔ HIGH DENIAL RISK' if prediction['risk_level']=='HIGH'
         else '⚠️ MEDIUM REVIEW RISK' if prediction['risk_level']=='MEDIUM'
         else '✅ LOW DENIAL RISK'}</b><br>
        <span style="color:#374151;">
        Score: {prediction['risk_score']:.4f} ·
        Review threshold: {prediction.get('review_threshold', 0):.4f} ·
        Denial threshold: {prediction.get('classification_threshold', 0):.4f}
        </span></div>""",
        unsafe_allow_html=True,
    )

    if features:
        with st.expander("Show model-ready features generated for this claim"):
            display = {
                FEATURE_LABELS.get(k, k): str(v)
                for k, v in features.items()
                if k in {
                    "diagnosis_code_missing", "procedure_code_missing", "billed_amount_missing",
                    "proc_no_diag", "diag_no_proc", "billed_deviation_imputed_capped",
                    "billed_amount_imputed", "log_billed_amount_imputed", "is_high_cost",
                    "cost_match_encoded", "provider_claim_count", "provider_violation_rate",
                    "patient_claim_count", "severity_rank", "specialty_encoded",
                }
            }
            st.dataframe(pd.DataFrame(list(display.items()), columns=["Feature", "Value"]),
                         use_container_width=True, hide_index=True)

    if explainer and features:
        explanation = explainer.explain(features, top_n=3)
        st.markdown("**Top 3 Denial Reasons (SHAP raw log-odds contributions)**")
        for reason in explanation["top_reasons"]:
            icon = "🔴" if reason["direction"] == "increases_risk" else "🟢"
            arrow = "↑ increases risk" if reason["direction"] == "increases_risk" else "↓ decreases risk"
            with st.expander(
                f"{icon} {reason['rank']}. {reason['label']} — SHAP {reason['shap_value']:+.4f} ({arrow})"
            ):
                st.write(f"**Feature:** `{reason['feature']}`")
                st.write(f"**How to fix:** {reason['fix']}")
                st.caption(explanation.get("note", "SHAP values are contributions, not probability percentages."))
    elif not explainer:
        st.info("SHAP explainer not available — run `python run_train.py` to generate/load the XGBoost model.")


def _render_model_ready_prediction(claim_features: dict, service: ClaimDenialService | None,
                                   explainer: SHAPExplainer | None) -> None:
    if service is None:
        st.error("Inference service could not be loaded. Run `python run_gold.py` and `python run_train.py`.")
        return
    prediction = service.predictor.predict(claim_features)
    _render_prediction_payload(prediction, claim_features, explainer)


def _unique_sorted(df: pd.DataFrame, col: str, include_missing: bool = True) -> list[str]:
    if df.empty or col not in df.columns:
        return [SENTINEL_MISSING] if include_missing else []
    vals = sorted(str(v) for v in df[col].dropna().unique() if str(v) != "")
    if include_missing and SENTINEL_MISSING not in vals:
        vals = [SENTINEL_MISSING] + vals
    return vals


def render_ml_tab(gold_dir: Path, models_dir: Path) -> None:
    st.header("ML Model — Performance, Explainability & Predictions")

    if not (models_dir / "training_report.json").exists():
        st.warning("Models not found. Train them first:", icon="⚠️")
        st.code("python run_gold.py\npython run_train.py", language="bash")
        return

    with st.spinner("Loading model, reports, and data..."):
        report = _load_report(models_dir)
        df = _load_gold(gold_dir)
        service = _load_service(gold_dir, models_dir)
        explainer = _load_explainer(models_dir)
        roc_data = _compute_roc(gold_dir, models_dir)
        cm_data = _compute_confusion(gold_dir, models_dir)

    st.subheader("1 — Model Performance")

    tuned = _metric_block(report)
    policy = _risk_policy(report)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Recommended", report.get("recommended_model", "—").replace("_", " ").title())
    c2.metric("ROC-AUC", f"{tuned.get('roc_auc', 0):.4f}")
    c3.metric("Recall", f"{tuned.get('recall', 0):.4f}", "primary")
    c4.metric("Precision", f"{tuned.get('precision', 0):.4f}")
    c5.metric("F1", f"{tuned.get('f1', 0):.4f}")

    st.caption(
        f"Thresholds are tuned, not hardcoded: review ≥ "
        f"{policy.get('medium_lower_inclusive', 0):.4f}, denial ≥ "
        f"{policy.get('classification_threshold', 0):.4f}."
    )

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

    st.divider()
    st.subheader("2 — Feature Intelligence")

    c_left, c_right = st.columns([3, 2])
    with c_left:
        if report.get("shap_importance"):
            st.plotly_chart(_shap_chart(report["shap_importance"]), key="ml_shap", width="stretch")
    with c_right:
        cal = report.get("calibration", {})
        st.markdown("**Calibration / probability quality**")
        st.metric("Brier Score", f"{cal.get('brier_score', 0):.4f}")
        st.metric("Expected Calibration Error", f"{cal.get('expected_calibration_error', 0):.4f}")
        st.caption("Lower is better. Use this before trusting scores as real probabilities.")

    if not df.empty:
        st.plotly_chart(_score_histogram(df, report), key="ml_hist", width="stretch")

    st.divider()
    st.subheader("3 — Prediction Explorer")
    st.caption("Pick a real Gold claim and score it using the currently recommended model + tuned threshold.")

    if df.empty:
        st.warning("Gold data not found. Run `python run_gold.py`.")
    else:
        features = report.get("features", [])
        claim_options = {}
        for _, row in df.head(5000).iterrows():
            label = (
                f"{row['claim_id']} | "
                f"{'⛔ Denied' if row['denial_flag'] == 1 else '✅ Approved'} | "
                f"diag={'✗' if row.get('diagnosis_code_missing') else '✓'} "
                f"proc={'✗' if row.get('procedure_code_missing') else '✓'} "
                f"amount={'✗' if row.get('billed_amount_missing') else '✓'}"
            )
            claim_options[label] = row.to_dict()

        selected_label = st.selectbox("Select a claim:", list(claim_options.keys()), key="claim_explorer_select")
        selected_claim = claim_options[selected_label]

        actual = "⛔ Denied" if selected_claim.get("denial_flag") == 1 else "✅ Approved"
        st.markdown(f"**Actual label:** {actual}")
        _render_model_ready_prediction(selected_claim, service, explainer)

        with st.expander("Show selected Gold feature values"):
            display = {
                FEATURE_LABELS.get(k, k): str(selected_claim.get(k))
                for k in features
            }
            st.dataframe(pd.DataFrame(list(display.items()), columns=["Feature", "Value"]),
                         use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("4 — Custom Claim Builder")
    st.caption(
        "Build a raw hypothetical claim. Derived flags, amount imputation, cost match, "
        "provider history, and risk features are computed automatically."
    )

    if service is None:
        st.error("Inference service unavailable. Run `python run_gold.py` and `python run_train.py`.")
        return

    provider_ids = _unique_sorted(df, "provider_id", include_missing=False) or ["UNKNOWN_PROVIDER"]
    diagnosis_codes = _unique_sorted(df, "diagnosis_code", include_missing=True)
    procedure_codes = _unique_sorted(df, "procedure_code", include_missing=True)
    specialties = _unique_sorted(df, "specialty", include_missing=False) or ["Unknown"]
    locations = _unique_sorted(df, "location", include_missing=False) or ["Unknown"]

    with st.form("custom_claim_form"):
        c1, c2, c3 = st.columns(3)
        claim_id = c1.text_input("Claim ID", value="CUSTOM")
        patient_id = c2.text_input("Patient ID", value="CUSTOM_PATIENT")
        provider_id = c3.selectbox("Provider ID", provider_ids)

        c1, c2, c3 = st.columns(3)
        diagnosis_code = c1.selectbox("Diagnosis Code", diagnosis_codes, index=1 if len(diagnosis_codes) > 1 else 0)
        procedure_code = c2.selectbox("Procedure Code", procedure_codes, index=1 if len(procedure_codes) > 1 else 0)
        amount_missing = c3.checkbox("Billed amount missing", value=False)

        billed_amount = None
        if not amount_missing:
            billed_amount = st.number_input(
                "Billed amount",
                min_value=0.0,
                max_value=100000.0,
                value=10000.0,
                step=500.0,
            )
        else:
            st.info("Amount will be treated as missing. Gold inference will impute it using procedure/global median.")

        c1, c2 = st.columns(2)
        specialty = c1.selectbox("Provider Specialty override", ["Use provider history"] + specialties)
        location = c2.selectbox("Provider Location override", ["Use provider history"] + locations)

        submitted = st.form_submit_button("🔍 Predict Denial Risk", use_container_width=True)

    if submitted:
        raw_claim = {
            "claim_id": claim_id,
            "patient_id": patient_id,
            "provider_id": provider_id,
            "diagnosis_code": None if diagnosis_code == SENTINEL_MISSING else diagnosis_code,
            "procedure_code": None if procedure_code == SENTINEL_MISSING else procedure_code,
            "billed_amount": None if amount_missing else billed_amount,
        }
        if specialty != "Use provider history":
            raw_claim["specialty"] = specialty
        if location != "Use provider history":
            raw_claim["location"] = location

        result = service.score_claim(raw_claim)
        st.markdown("---")
        st.markdown("**Prediction for custom claim:**")
        if result["status"] == "success":
            _render_prediction_payload(result["prediction"], result["features"], explainer)
        else:
            err = result["error"]
            st.error(
                f"{err['error_code']}: {err['message']} "
                f"(occurrence_count={err['occurrence_count']}, repeated={err['is_repeated']})"
            )

    with st.expander("Training details"):
        split = report.get("split_policy", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train rows", split.get("train_rows", 0))
        c2.metric("Validation rows", split.get("validation_rows", 0))
        c3.metric("Test rows", split.get("test_rows", 0))
        c4.metric("Features", report.get("feature_count", 0))
        st.caption(
            f"Recommended model: **{report.get('recommended_model', '—').upper()}** · "
            "Threshold tuned on validation set; final metrics reported on held-out test set."
        )
