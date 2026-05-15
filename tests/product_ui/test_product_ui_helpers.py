from product_ui.rendering import dedupe_policy_evidence, short_text


def test_short_text_truncates_without_newlines():
    text = "A" * 300 + "\nB"
    out = short_text(text, limit=40)
    assert len(out) <= 40
    assert "\n" not in out
    assert out.endswith("…")


def test_dedupe_policy_evidence_keeps_best_score():
    rows = [
        {"reason_code": "A", "source_name": "s.md", "section_title": "x", "similarity_score": 0.5, "policy_summary": "old"},
        {"reason_code": "A", "source_name": "s.md", "section_title": "x", "similarity_score": 0.9, "policy_summary": "new"},
        {"reason_code": "B", "source_name": "s.md", "section_title": "x", "similarity_score": 0.4, "policy_summary": "other"},
    ]
    out = dedupe_policy_evidence(rows)
    assert len(out) == 2
    assert out[0]["policy_summary"] == "new"
    assert out[0]["similarity_score"] == 0.9
