from pathlib import Path

from src.observability import ErrorCode, ErrorTracker
from src.observability.exceptions import ClaimDenialError


def test_error_tracker_records_repeated_errors(tmp_path: Path):
    tracker = ErrorTracker(log_dir=tmp_path)

    event1 = tracker.record(
        ErrorCode.INFER_INVALID_CLAIM,
        "Invalid custom claim",
        component="inference",
        stage="custom_claim_builder",
        metadata={"stage": "custom_claim_builder", "field": "billed_amount"},
    )
    event2 = tracker.record(
        ErrorCode.INFER_INVALID_CLAIM,
        "Invalid custom claim",
        component="inference",
        stage="custom_claim_builder",
        metadata={"stage": "custom_claim_builder", "field": "billed_amount"},
    )

    assert event1.occurrence_count == 1
    assert event2.occurrence_count == 2
    assert event2.is_repeated is True
    assert (tmp_path / "error_events.jsonl").exists()
    assert (tmp_path / "error_summary.json").exists()

    repeated = tracker.get_repeated_errors()
    assert len(repeated) == 1
    assert repeated[0]["error_code"] == ErrorCode.INFER_INVALID_CLAIM.value
    assert repeated[0]["count"] == 2


def test_error_tracker_preserves_claim_denial_error_code(tmp_path: Path):
    tracker = ErrorTracker(log_dir=tmp_path)
    exc = ClaimDenialError(
        ErrorCode.GOLD_ROW_COUNT_MISMATCH,
        "Gold row count changed after joins",
        component="gold",
        metadata={"stage": "build_base", "table": "gold_claim_base"},
    )

    event = tracker.record_exception(exc, component="gold", stage="build_base")

    assert event.error_code == ErrorCode.GOLD_ROW_COUNT_MISMATCH.value
    assert event.component == "gold"
    assert event.stage == "build_base"
    assert event.metadata["table"] == "gold_claim_base"
