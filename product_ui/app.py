"""Role-aware Streamlit product UI for Claim Denial Prevention.

Run after FastAPI is running:
    streamlit run product_ui/app.py

Roles:
- analyst: business-focused analytics + custom claim workflow
- developer: technical dashboard + custom claim workflow
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dev_dashboard"))

from product_ui.api_client import ApiClientError, ClaimDenialApiClient, DEFAULT_API_BASE_URL
from product_ui.rendering import (
    artifact_health_counts,
    artifact_rows_from_payload,
    dedupe_policy_evidence,
    deployment_readiness_rows,
    overall_health_label,
    risk_badge_text,
    short_text,
    visible_tabs_for_role,
)

BRONZE_DIR = ROOT / "data" / "bronze"
SILVER_DIR = ROOT / "data" / "silver"
GOLD_DIR = ROOT / "data" / "gold"
MODELS_DIR = ROOT / "models"
VECTOR_DIR = ROOT / "data" / "vector_store"
POLICY_PROCESSED_DIR = ROOT / "data" / "policies" / "processed"

st.set_page_config(
    page_title="Claim Denial Prevention",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def _read_parquet(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data(show_spinner=False)
def _read_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _client() -> ClaimDenialApiClient:
    return ClaimDenialApiClient(token=st.session_state.get("access_token"))


def _init_session() -> None:
    st.session_state.setdefault("access_token", None)
    st.session_state.setdefault("user", None)
    st.session_state.setdefault("last_claim_result", None)
    st.session_state.setdefault("overview_claim_result", None)
    st.session_state.setdefault("last_validation_result", None)


def _logout() -> None:
    st.session_state["access_token"] = None
    st.session_state["user"] = None
    st.session_state["last_claim_result"] = None
    st.session_state["overview_claim_result"] = None
    st.session_state["last_validation_result"] = None
    st.rerun()


def _render_login() -> None:
    st.title("Claim Denial Prevention")
    st.caption("AI-powered risk prediction, policy evidence, and remediation guidance")

    left, right = st.columns([0.9, 1.1], vertical_alignment="center")
    with left:
        st.subheader("Sign in")
        with st.form("login_form"):
            email = st.text_input("Email", value="analyst@example.com", key="login_email")
            password = st.text_input("Password", type="password", value="analyst12345", key="login_password")
            submitted = st.form_submit_button("Login", width="stretch")
        if submitted:
            try:
                payload = ClaimDenialApiClient().login(email=email, password=password)
                st.session_state["access_token"] = payload["access_token"]
                st.session_state["user"] = payload["user"]
                st.success("Login successful")
                st.rerun()
            except ApiClientError as exc:
                st.error(str(exc))

    with right:
        st.markdown("### What this product does")
        st.write(
            "Billing teams can check a claim before submission. The system validates the input, "
            "predicts denial risk, explains the drivers, retrieves policy evidence, and suggests remediation steps."
        )
        st.info(f"API endpoint: `{DEFAULT_API_BASE_URL}`")
        st.markdown("**Demo users**")
        st.code("analyst@example.com / analyst12345\ndeveloper@example.com / dev12345", language="text")


def _render_sidebar() -> None:
    user = st.session_state.get("user") or {}
    with st.sidebar:
        st.markdown("### Claim Denial Prevention")
        st.caption("Role-aware product UI")
        st.divider()
        st.markdown(f"**Signed in as:** `{user.get('email')}`")
        st.markdown(f"**Role:** `{user.get('role')}`")
        st.button("Logout", on_click=_logout, width="stretch")
        st.divider()
        try:
            health = ClaimDenialApiClient().health()
            st.success(f"API: {health.get('status', 'ok')}")
        except ApiClientError:
            st.error("API unavailable")


def _metric_snapshot() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    features = _read_parquet(str(GOLD_DIR / "gold_claim_features.parquet"))
    final = _read_parquet(str(GOLD_DIR / "gold_claim_final_explanations.parquet"))
    training = _read_json(str(MODELS_DIR / "training_report.json"))
    return features, final, training


def _render_overview_metrics(features: pd.DataFrame, final: pd.DataFrame, training: dict[str, Any]) -> None:
    cols = st.columns(4)
    cols[0].metric("Claims", f"{len(features):,}" if not features.empty else "Missing")
    if not features.empty and "denial_flag" in features:
        cols[1].metric("Historical denial rate", f"{features['denial_flag'].mean():.1%}")
    else:
        cols[1].metric("Historical denial rate", "n/a")
    cols[2].metric("Final explanations", f"{len(final):,}" if not final.empty else "Missing")
    cols[3].metric("Model", training.get("recommended_model", "n/a"))


def _render_developer_overview_details(features: pd.DataFrame, final: pd.DataFrame, training: dict[str, Any]) -> None:
    st.markdown("### Data and model readiness")
    left, right = st.columns([1, 1])
    with left:
        artifact_rows = [
            {"artifact": "Gold features", "status": "Available" if not features.empty else "Missing"},
            {"artifact": "Final explanations", "status": "Available" if not final.empty else "Missing"},
            {"artifact": "Training report", "status": "Available" if training else "Missing"},
            {"artifact": "Vector metadata", "status": "Available" if (VECTOR_DIR / "policy_metadata.json").exists() else "Missing"},
            {"artifact": "Policy chunks", "status": "Available" if (POLICY_PROCESSED_DIR / "policy_chunks.parquet").exists() else "Missing"},
        ]
        st.dataframe(pd.DataFrame(artifact_rows), width="stretch", hide_index=True)
    with right:
        if not features.empty and "denial_flag" in features:
            st.markdown("**Claim outcome distribution**")
            dist = features["denial_flag"].map({0: "not denied", 1: "denied"}).fillna("unknown").value_counts().reset_index()
            dist.columns = ["outcome", "claim_count"]
            st.bar_chart(dist, x="outcome", y="claim_count")
        else:
            st.info("Run the feature pipeline to see outcome distribution.")

    if not features.empty:
        st.markdown("### Quick data preview")
        preview_cols = [
            c for c in [
                "claim_id",
                "provider_id",
                "diagnosis_code",
                "procedure_code",
                "billed_amount",
                "billed_amount_imputed",
                "denial_flag",
            ] if c in features.columns
        ]
        if preview_cols:
            st.dataframe(features[preview_cols].head(12), width="stretch", hide_index=True)


def _render_analyst_overview_details(features: pd.DataFrame) -> None:
    st.markdown("### Business workflow")
    st.write("Use the pre-submission check below for a claim, then open **Claim Analytics** for high-level claim trends.")

    if not features.empty:
        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Top provider specialties**")
            if "specialty" in features.columns:
                specialty = features["specialty"].fillna("Unknown").value_counts().head(6).reset_index()
                specialty.columns = ["specialty", "claim_count"]
                st.bar_chart(specialty, x="specialty", y="claim_count")
            else:
                st.info("Specialty column is unavailable.")
        with right:
            st.markdown("**Billing amount snapshot**")
            amount_col = "billed_amount" if "billed_amount" in features.columns else "billed_amount_imputed"
            if amount_col in features.columns:
                amounts = pd.to_numeric(features[amount_col], errors="coerce").dropna()
                if not amounts.empty:
                    q50 = amounts.median()
                    q90 = amounts.quantile(0.90)
                    st.metric("Median billed amount", f"{q50:,.0f}")
                    st.metric("90th percentile", f"{q90:,.0f}")
                else:
                    st.info("No numeric amount values available.")
            else:
                st.info("Amount column is unavailable.")

    st.divider()
    st.markdown("### Quick custom claim check")
    st.caption("This is the main analyst workflow: validate a claim, estimate risk, retrieve policy evidence, and generate remediation steps.")
    _render_custom_claim(
        form_key="overview_custom_claim_form",
        result_key="overview_claim_result",
        title="Quick Custom Claim",
        compact=True,
    )


def _render_overview(role: str) -> None:
    st.subheader("Overview")
    features, final, training = _metric_snapshot()
    _render_overview_metrics(features, final, training)

    if role == "analyst":
        _render_analyst_overview_details(features)
    else:
        st.markdown("### Developer workflow")
        st.write("Inspect data layers, model behavior, explanations, policy retrieval, and artifact health before cloud deployment.")
        _render_developer_overview_details(features, final, training)


def _render_claim_analytics() -> None:
    st.subheader("Claim Analytics")
    features = _read_parquet(str(GOLD_DIR / "gold_claim_features.parquet"))
    if features.empty:
        st.warning("Gold feature table is missing. Run the pipeline first.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Total claims", f"{len(features):,}")
    if "denial_flag" in features:
        c2.metric("Denied claims", f"{int(features['denial_flag'].sum()):,}")
        c3.metric("Denial rate", f"{features['denial_flag'].mean():.1%}")
    else:
        c2.metric("Denied claims", "n/a")
        c3.metric("Denial rate", "n/a")

    chart_cols = [c for c in ["specialty", "location", "severity", "cost_match_level"] if c in features.columns]
    if chart_cols:
        selected = st.selectbox("Group claims by", chart_cols, key="analyst_group_by")
        counts = features[selected].fillna("Unknown").value_counts().reset_index()
        counts.columns = [selected, "claim_count"]
        st.bar_chart(counts, x=selected, y="claim_count")

    display_cols = [c for c in ["claim_id", "provider_id", "diagnosis_code", "procedure_code", "billed_amount", "denial_flag"] if c in features.columns]
    st.dataframe(features[display_cols].head(100), width="stretch", hide_index=True)


def _claim_form(*, form_key: str, title: str = "Custom Claim", compact: bool = False) -> dict[str, Any] | None:
    """Render a custom-claim form.

    The form key must be supplied by callers because Streamlit renders all tabs
    eagerly. Unique keys prevent collisions with both this product UI and the
    embedded developer dashboard's Week 4 custom claim builder.
    """
    with st.form(form_key):
        st.markdown(f"### {title}")
        c1, c2, c3 = st.columns(3)
        claim_id = c1.text_input("Claim ID", value="CUSTOM001", key=f"{form_key}_claim_id")
        patient_id = c2.text_input("Patient ID", value="P001", key=f"{form_key}_patient_id")
        provider_id = c3.text_input("Provider ID", value="PR100", key=f"{form_key}_provider_id")

        c4, c5, c6 = st.columns(3)
        diagnosis_code = c4.text_input("Diagnosis Code", value="", key=f"{form_key}_diagnosis_code")
        procedure_code = c5.text_input("Procedure Code", value="PROC1", key=f"{form_key}_procedure_code")
        billed_amount_raw = c6.text_input("Billed Amount", value="12000", key=f"{form_key}_billed_amount")

        if compact:
            specialty = ""
            location = ""
        else:
            c7, c8 = st.columns(2)
            specialty = c7.text_input("Specialty (optional)", value="", key=f"{form_key}_specialty")
            location = c8.text_input("Location (optional)", value="", key=f"{form_key}_location")

        submitted = st.form_submit_button("Check Claim", width="stretch")

    if not submitted:
        return None

    billed_amount: float | None
    if billed_amount_raw.strip() == "":
        billed_amount = None
    else:
        try:
            billed_amount = float(billed_amount_raw)
        except ValueError:
            st.error("Billed Amount must be numeric or blank.")
            return None

    return {
        "claim_id": claim_id,
        "patient_id": patient_id,
        "provider_id": provider_id,
        "diagnosis_code": diagnosis_code or None,
        "procedure_code": procedure_code or None,
        "billed_amount": billed_amount,
        "specialty": specialty or None,
        "location": location or None,
    }


def _render_validation(validation: dict[str, Any]) -> None:
    blocking = validation.get("blocking_errors") or []
    warnings = validation.get("warnings") or []
    infos = validation.get("infos") or []
    if blocking:
        st.error("Blocking input errors must be fixed before scoring.")
        for item in blocking:
            st.write(f"- **{item.get('field') or item.get('code')}**: {item.get('message')}")
    if warnings:
        st.warning("Non-blocking validation warnings")
        for item in warnings:
            st.write(f"- **{item.get('field') or item.get('code')}**: {item.get('message')}")
    if infos and not blocking and not warnings:
        st.success("Input passed validation.")



def _render_policy_evidence_compact(evidence: list[dict[str, Any]]) -> None:
    if not evidence:
        return

    items = dedupe_policy_evidence(evidence)
    st.markdown("### Policy support")
    st.caption("Top supporting policy sections are summarized here. Open details for full retrieved evidence.")

    top = items[:3]
    for item in top:
        source = item.get("source_name") or "policy source"
        section = item.get("section_title") or "policy section"
        score = item.get("similarity_score")
        label = f"{source} · {section}"
        if score is not None:
            try:
                label = f"{label} · score {float(score):.2f}"
            except (TypeError, ValueError):
                pass
        with st.container(border=True):
            st.markdown(f"**{label}**")
            st.write(short_text(item.get("policy_summary"), limit=260))

    with st.expander(f"Detailed policy evidence ({len(items)} retrieved sections)"):
        for idx, item in enumerate(items, start=1):
            source = item.get("source_name") or "policy source"
            section = item.get("section_title") or "policy section"
            reason = item.get("reason_code") or "reason"
            score = item.get("similarity_score")
            st.markdown(f"**{idx}. {reason} — {source} / {section}**")
            if score is not None:
                try:
                    st.caption(f"Similarity score: {float(score):.3f}")
                except (TypeError, ValueError):
                    pass
            st.write(item.get("policy_summary"))
            st.divider()


def _render_claim_result(result: dict[str, Any]) -> None:
    envelope_status = result.get("status")
    data = result.get("data") if "data" in result else result
    if not isinstance(data, dict):
        st.error("Unexpected API response format.")
        st.json(result)
        return

    prediction = data.get("prediction") or {}
    decision = data.get("decision") or {}
    presentation = data.get("agent_presentation") or {}

    st.markdown("### Result")
    cols = st.columns(4)
    cols[0].metric("Risk", risk_badge_text(prediction))
    cols[1].metric("Decision", decision.get("status", envelope_status or "n/a"))
    cols[2].metric("Priority", decision.get("priority", "n/a"))
    cols[3].metric("Presentation", presentation.get("source", "deterministic"))

    if decision.get("summary"):
        st.info(decision.get("summary"))

    validation = data.get("validation") or {}
    with st.expander("Validation details", expanded=bool(validation.get("blocking_errors") or validation.get("warnings"))):
        _render_validation(validation)

    st.markdown("### Recommended action plan")
    actions = presentation.get("action_plan") or [item.get("action") for item in data.get("recommendations", []) if item.get("action")]
    if actions:
        for idx, action in enumerate(actions, start=1):
            st.write(f"{idx}. {action}")
    else:
        st.write("No action required beyond standard claim checks.")

    reasons = data.get("reasons") or []
    if reasons:
        st.markdown("### Risk reasons")
        for reason in reasons:
            with st.container(border=True):
                st.markdown(f"**{reason.get('reason_title') or reason.get('reason_code')}**")
                st.write(reason.get("reason_text"))
                if reason.get("fix_suggestion"):
                    st.caption(f"Suggested fix: {reason.get('fix_suggestion')}")
    else:
        st.success("No denial-risk reasons were generated for this claim. Review validation warnings, if any.")

    _render_policy_evidence_compact(data.get("policy_evidence") or [])

    notes = presentation.get("analyst_notes") or []
    if notes:
        with st.expander("Analyst notes"):
            for note in notes:
                st.write(f"- {note}")

    with st.expander("Raw API response"):
        st.json(result)


def _render_custom_claim(*, form_key: str = "product_custom_claim_form", result_key: str = "last_claim_result", title: str = "Custom Claim", compact: bool = False) -> None:
    st.subheader("Check Custom Claim")
    st.caption("This form calls FastAPI, which runs validation, ML prediction, explanations, policy retrieval, and the remediation agent.")
    claim = _claim_form(form_key=form_key, title=title, compact=compact)
    if claim:
        try:
            with st.spinner("Analyzing claim..."):
                result = _client().recommend_claim(claim)
            st.session_state[result_key] = result
        except ApiClientError as exc:
            st.error(str(exc))

    if st.session_state.get(result_key):
        _render_claim_result(st.session_state[result_key])


def _health_payloads() -> tuple[bool, dict[str, Any], dict[str, Any] | None, str | None]:
    """Load public and protected health payloads for the current UI session."""
    api_ok = False
    public_health: dict[str, Any] = {}
    artifact_health: dict[str, Any] | None = None
    error: str | None = None
    try:
        public_health = ClaimDenialApiClient().health()
        api_ok = str(public_health.get("status", "")).lower() == "ok"
    except ApiClientError as exc:
        error = str(exc)

    if st.session_state.get("access_token"):
        try:
            artifact_health = _client().artifact_health()
        except ApiClientError as exc:
            error = str(exc)
    return api_ok, public_health, artifact_health, error


def _render_status_card(title: str, value: str, detail: str = "") -> None:
    with st.container(border=True):
        st.caption(title)
        st.markdown(f"### {value}")
        if detail:
            st.write(detail)


def _render_system_health(role: str) -> None:
    st.subheader("System Health")
    api_ok, public_health, artifact_health, error = _health_payloads()
    counts = artifact_health_counts(artifact_health)
    label = overall_health_label(api_ok, artifact_health)

    if role == "analyst":
        c1, c2, c3 = st.columns(3)
        with c1:
            _render_status_card("Overall status", label, "Claim checking is available." if label == "Ready" else "Some services need developer attention.")
        with c2:
            _render_status_card("Claim checker", "Available" if api_ok else "Unavailable", "FastAPI is reachable." if api_ok else "The backend could not be reached.")
        with c3:
            ready_txt = f"{counts['ready']}/{counts['total']} ready" if counts["total"] else "Not checked"
            _render_status_card("AI artifacts", ready_txt, "Model and policy assets used by recommendations.")

        if label == "Ready":
            st.success("The claim review system is ready. Use **Custom Claim** to validate and review a claim before submission.")
        elif error:
            st.warning("The system is partially available. If claim checking fails, contact the developer/admin.")
        return

    # Developer view: business-readable cards first; raw JSON is hidden below.
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _render_status_card("API", "OK" if api_ok else "Down", public_health.get("service", "claim-denial-api"))
    with c2:
        _render_status_card("Artifacts", f"{counts['ready']}/{counts['total']} ready" if counts["total"] else "Not checked", "Model/RAG runtime files")
    with c3:
        openai_enabled = str(_read_json(str(ROOT / ".runtime_flags.json")).get("openai", "disabled"))
        _render_status_card("OpenAI layer", "Optional", "Disabled locally unless configured")
    with c4:
        _render_status_card("Auth DB", "SQLite local", "RDS PostgreSQL during AWS deployment")

    if error:
        st.warning(error)

    artifact_rows = artifact_rows_from_payload(artifact_health)
    if artifact_rows:
        st.markdown("### Artifact readiness")
        st.dataframe(pd.DataFrame(artifact_rows), width="stretch", hide_index=True)

    st.markdown("### Deployment readiness")
    st.dataframe(pd.DataFrame(deployment_readiness_rows(api_ok=api_ok, artifact_payload=artifact_health)), width="stretch", hide_index=True)

    st.markdown("### Local artifact paths")
    rows = []
    for name, path in {
        "Gold features": GOLD_DIR / "gold_claim_features.parquet",
        "Inference artifacts": GOLD_DIR / "inference_artifacts.json",
        "Training report": MODELS_DIR / "training_report.json",
        "XGBoost model": MODELS_DIR / "xgb_model.pkl",
        "Vector metadata": VECTOR_DIR / "policy_metadata.json",
        "Policy chunks": POLICY_PROCESSED_DIR / "policy_chunks.parquet",
    }.items():
        rows.append({"artifact": name, "path": str(path), "exists": path.exists()})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with st.expander("Developer details: raw health payloads"):
        st.markdown("**Public health**")
        st.json(public_health)
        st.markdown("**Protected artifact health**")
        st.json(artifact_health or {})


def _render_risk_model_summary_for_product() -> None:
    """Product-friendly developer summary before the embedded ML dashboard."""
    report = _read_json(str(MODELS_DIR / "training_report.json"))
    if not report:
        st.warning("Training report is missing. Run `python run_train.py` before reviewing model governance.")
        return

    tuned = report.get("recommended_model_test_at_tuned_threshold") or {}
    policy = report.get("risk_band_policy") or {}
    calibration = report.get("calibration") or {}

    st.markdown("### Model governance snapshot")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Model", str(report.get("recommended_model", "n/a")).replace("_", " ").title())
    c2.metric("ROC-AUC", f"{float(tuned.get('roc_auc', 0) or 0):.3f}")
    c3.metric("Recall", f"{float(tuned.get('recall', 0) or 0):.3f}")
    c4.metric("Precision", f"{float(tuned.get('precision', 0) or 0):.3f}")
    c5.metric("Features", report.get("feature_count", len(report.get("features", []) or [])))

    t1, t2, t3 = st.columns(3)
    t1.metric("Review threshold", f"{float(policy.get('medium_lower_inclusive', 0) or 0):.3f}")
    t2.metric("Denial threshold", f"{float(policy.get('classification_threshold', 0) or 0):.3f}")
    t3.metric("Brier score", f"{float(calibration.get('brier_score', 0) or 0):.3f}")
    st.caption("This section summarizes model behavior before the detailed internal Risk Model dashboard below.")


def _render_risk_model_artifact_summary() -> None:
    st.markdown("### Model artifacts and feature contract")
    rows = []
    for name, path in {
        "XGBoost model": MODELS_DIR / "xgb_model.pkl",
        "Logistic model": MODELS_DIR / "lr_model.pkl",
        "Training report": MODELS_DIR / "training_report.json",
        "Threshold report": MODELS_DIR / "threshold_report.json",
        "Calibration report": MODELS_DIR / "calibration_report.json",
        "Model card": MODELS_DIR / "model_card.json",
        "Inference artifacts": GOLD_DIR / "inference_artifacts.json",
    }.items():
        rows.append({"artifact": name, "path": str(path), "exists": path.exists()})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    report = _read_json(str(MODELS_DIR / "training_report.json"))
    features = report.get("features") or []
    if features:
        with st.expander(f"Feature contract ({len(features)} model features)"):
            st.dataframe(pd.DataFrame({"feature": features}), width="stretch", hide_index=True)


def _render_developer_tab(tab_name: str) -> None:
    try:
        if tab_name == "Data Pipeline":
            from tabs.raw_data import render_raw_tab
            from tabs.clean_data import render_clean_tab
            sub_raw, sub_clean = st.tabs(["Raw Data", "Clean Data"])
            with sub_raw:
                render_raw_tab(bronze_dir=BRONZE_DIR)
            with sub_clean:
                render_clean_tab(bronze_dir=BRONZE_DIR, silver_dir=SILVER_DIR)
        elif tab_name == "Risk Model":
            from tabs.ml_analysis import render_ml_tab
            _render_risk_model_summary_for_product()
            st.divider()
            render_ml_tab(gold_dir=GOLD_DIR, models_dir=MODELS_DIR)
            st.divider()
            _render_risk_model_artifact_summary()
        elif tab_name == "Risk Explanations":
            from tabs.explainability import render_explainability_tab
            render_explainability_tab(gold_dir=GOLD_DIR, models_dir=MODELS_DIR)
        elif tab_name == "Policy Evidence":
            from tabs.policy_rag import render_policy_rag_tab
            render_policy_rag_tab(root_dir=ROOT, gold_dir=GOLD_DIR, models_dir=MODELS_DIR)
        elif tab_name == "Retrieval Analytics":
            from tabs.retrieval_analytics import render_retrieval_analytics_tab
            render_retrieval_analytics_tab(root_dir=ROOT, gold_dir=GOLD_DIR, models_dir=MODELS_DIR)
        else:
            st.info("Developer tab is not available.")
    except Exception as exc:
        st.error(f"Could not render developer tab: {exc}")
        st.caption("The original dev dashboard remains available with: streamlit run dev_dashboard/app.py")


def _render_authenticated() -> None:
    user = st.session_state.get("user") or {}
    role = str(user.get("role") or "analyst").lower()
    _render_sidebar()

    st.title("Claim Denial Prevention")
    st.caption(f"Role-aware product UI · {role}")

    tab_names = visible_tabs_for_role(role)
    tabs = st.tabs(tab_names)
    for tab, name in zip(tabs, tab_names):
        with tab:
            if name == "Overview":
                _render_overview(role)
            elif name == "Claim Analytics":
                _render_claim_analytics()
            elif name == "Custom Claim":
                _render_custom_claim(form_key="product_custom_claim_form", result_key="last_claim_result", title="Custom Claim", compact=False)
            elif name == "System Health":
                _render_system_health(role)
            else:
                _render_developer_tab(name)


def main() -> None:
    _init_session()
    if not st.session_state.get("access_token") or not st.session_state.get("user"):
        _render_login()
    else:
        _render_authenticated()


if __name__ == "__main__":
    main()
