import json
from pathlib import Path


def test_policy_source_registry_has_required_source_categories():
    root = Path(__file__).resolve().parents[2]
    path = root / "policy_docs" / "official_policy_source_registry.json"
    registry = json.loads(path.read_text())
    source_types = {s["source_type"] for s in registry["sources"]}
    assert "official_government" in source_types
    assert "public_payer_reference" in source_types
    assert "internal_project_policy" in source_types


def test_policy_source_registry_has_week6_required_tags():
    root = Path(__file__).resolve().parents[2]
    path = root / "policy_docs" / "official_policy_source_registry.json"
    registry = json.loads(path.read_text())
    all_tags = {tag for source in registry["sources"] for tag in source.get("tags", [])}
    for tag in ["ncci", "prior_authorization", "medical_necessity", "hipaa", "security", "payer"]:
        assert tag in all_tags


def test_curated_policy_docs_exist():
    root = Path(__file__).resolve().parents[2]
    seed_dir = root / "policy_docs" / "rag_seed"
    expected = {
        "us_healthcare_claim_policy_seed_pack.md",
        "payer_policy_reference_pack.md",
        "hipaa_operational_safeguards_pack.md",
    }
    assert expected.issubset({p.name for p in seed_dir.glob("*.md")})
