from product_ui.rendering import analyst_result_sections, risk_badge_text, visible_tabs_for_role


def test_visible_tabs_are_role_specific():
    assert "Retrieval Analytics" in visible_tabs_for_role("developer")
    assert "Retrieval Analytics" not in visible_tabs_for_role("analyst")
    assert "Custom Claim" in visible_tabs_for_role("analyst")


def test_risk_badge_text_formats_score():
    assert risk_badge_text({"risk_level": "HIGH", "risk_score": 0.9937}) == "HIGH · 99.4%"
    assert risk_badge_text(None) == "Risk unavailable"


def test_analyst_result_sections_counts_nested_output():
    result = {
        "status": "success",
        "data": {
            "validation": {"warnings": [{"code": "W"}]},
            "reasons": [{"reason_code": "A"}, {"reason_code": "B"}],
            "policy_evidence": [{"source_name": "policy.md"}],
            "recommendations": [{"action": "Review"}],
            "agent_presentation": {"action_plan": ["Review"]},
        },
    }
    assert analyst_result_sections(result) == {
        "warnings": 1,
        "reasons": 2,
        "policy_evidence": 1,
        "recommendations": 1,
        "action_plan": 1,
    }
