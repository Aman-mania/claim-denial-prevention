"""Optional OpenAI presentation layer for the remediation agent.

OpenAI is used only as the final readability/presentation layer. The source of
truth remains deterministic: validation issues, ML prediction, SHAP/business
reasons, RAG policy evidence, and recommendation catalog actions.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from src.agent.schemas import AgentPresentation

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(?P<body>.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _strip_json_block(text: str) -> str:
    match = _JSON_BLOCK_RE.search(text or "")
    return match.group("body").strip() if match else (text or "").strip()


def deterministic_presentation(agent_payload: dict[str, Any], *, source: str = "deterministic") -> dict[str, Any]:
    """Create a stable readable presentation without external LLM calls."""
    prediction = agent_payload.get("prediction") or {}
    validation = agent_payload.get("validation") or {}
    decision = agent_payload.get("decision") or {}
    reasons = agent_payload.get("reasons") or []
    evidence = agent_payload.get("policy_evidence") or []
    recommendations = agent_payload.get("recommendations") or []

    claim_id = agent_payload.get("claim_id") or prediction.get("claim_id") or "the claim"
    risk_level = prediction.get("risk_level") or "UNKNOWN"
    risk_score = prediction.get("risk_score")
    if isinstance(risk_score, (float, int)):
        score_text = f"{risk_score:.1%}" if risk_score <= 1 else f"{risk_score:.2f}"
    else:
        score_text = "not available"

    if validation.get("blocking_errors"):
        summary = f"Claim {claim_id} is blocked because required input validation failed. Correct the blocking fields before scoring or submission."
    else:
        summary = f"Claim {claim_id} is {risk_level} risk with a score of {score_text}. {decision.get('summary', '')}".strip()

    action_plan = [str(item.get("action")) for item in recommendations[:5] if item.get("action")]
    analyst_notes = []
    for reason in reasons[:3]:
        title = reason.get("reason_title") or reason.get("reason_code") or "Risk reason"
        text = reason.get("reason_text") or ""
        analyst_notes.append(f"{title}: {text}".strip())
    for policy in evidence[:3]:
        src = policy.get("source_name") or "policy source"
        summary_text = policy.get("policy_summary") or policy.get("policy_text") or ""
        if summary_text:
            analyst_notes.append(f"Policy evidence from {src}: {summary_text}")

    caveats = [
        "This response summarizes deterministic ML, explainability, policy retrieval, and recommendation outputs.",
        "OpenAI/LLM text, when enabled, is a presentation layer only and should not override structured fields.",
    ]

    return AgentPresentation(
        source=source,
        summary=summary,
        action_plan=action_plan,
        analyst_notes=analyst_notes[:6],
        caveats=caveats,
    ).to_dict()


class OpenAIPresentationLayer:
    """Generate structured analyst-readable output using OpenAI when enabled."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.enabled = _bool_env("ENABLE_OPENAI_AGENT_OUTPUT", False) if enabled is None else bool(enabled)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_AGENT_MODEL", "gpt-4o-mini")
        self.timeout_seconds = float(timeout_seconds or os.getenv("OPENAI_AGENT_TIMEOUT_SECONDS", "20"))

    def generate(self, agent_payload: dict[str, Any]) -> dict[str, Any]:
        """Return presentation JSON. Falls back deterministically on any error."""
        if not self.enabled or not self.api_key:
            return deterministic_presentation(agent_payload, source="deterministic")

        try:
            return self._generate_with_openai(agent_payload)
        except Exception as exc:
            fallback = deterministic_presentation(agent_payload, source="deterministic_fallback")
            fallback["caveats"].append(f"OpenAI presentation failed and deterministic fallback was used: {exc}")
            return fallback

    def _generate_with_openai(self, agent_payload: dict[str, Any]) -> dict[str, Any]:
        """Call OpenAI SDK and parse a strict JSON object from the model output."""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, timeout=self.timeout_seconds)
        system = (
            "You are the final presentation layer for a healthcare claim denial-prevention system. "
            "Use only the provided structured facts. Do not invent policy, scores, reasons, or recommendations. "
            "Return valid JSON only with keys: summary, action_plan, analyst_notes, caveats. "
            "action_plan, analyst_notes, and caveats must be arrays of short strings. "
            "The tone should be clear, professional, and suitable for a billing analyst demo."
        )
        user = json.dumps({
            "task": "Rewrite the deterministic agent result into a readable structured response without changing facts.",
            "agent_payload": agent_payload,
        }, default=str)

        text = ""
        if hasattr(client, "responses"):
            response = client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            text = getattr(response, "output_text", "") or ""
        else:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
            )
            text = response.choices[0].message.content or ""

        parsed = json.loads(_strip_json_block(text))
        if not isinstance(parsed, dict):
            raise ValueError("OpenAI response was not a JSON object")
        return AgentPresentation(
            source="openai",
            summary=str(parsed.get("summary") or ""),
            action_plan=[str(x) for x in parsed.get("action_plan", []) if str(x).strip()],
            analyst_notes=[str(x) for x in parsed.get("analyst_notes", []) if str(x).strip()],
            caveats=[str(x) for x in parsed.get("caveats", []) if str(x).strip()],
        ).to_dict()
