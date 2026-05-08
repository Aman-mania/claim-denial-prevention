from pathlib import Path

from src.observability.error_codes import ErrorCode
from src.observability.error_tracker import ErrorTracker
from src.observability.pipeline_integration import record_pipeline_report, summarize_error_events


def test_records_failed_dataset_report(tmp_path: Path):
    tracker = ErrorTracker(log_dir=tmp_path)
    report = {
        "datasets": {
            "claims": {
                "status": "failed",
                "error": "Raw CSV not found: data/raw/claims_1000.csv",
            }
        }
    }

    events = record_pipeline_report(report, component="ingestion", tracker=tracker, stage="run_ingestion")

    assert len(events) == 1
    assert events[0].error_code == ErrorCode.INGEST_RAW_FILE_MISSING.value
    assert events[0].occurrence_count == 1


def test_repeated_errors_are_detected(tmp_path: Path):
    tracker = ErrorTracker(log_dir=tmp_path, repeat_alert_threshold=2)
    report = {
        "datasets": {
            "claims": {
                "status": "failed",
                "error": "Raw CSV not found: data/raw/claims_1000.csv",
            }
        }
    }

    first = record_pipeline_report(report, component="ingestion", tracker=tracker, stage="run_ingestion")
    second = record_pipeline_report(report, component="ingestion", tracker=tracker, stage="run_ingestion")

    assert first[0].is_repeated is False
    assert second[0].is_repeated is True
    assert second[0].occurrence_count == 2


def test_records_validation_warning(tmp_path: Path):
    tracker = ErrorTracker(log_dir=tmp_path)
    report = {
        "datasets": {
            "claims": {
                "status": "success",
                "validation": {"status": "warnings", "errors": [{"column": "denial_flag"}]},
            }
        }
    }

    events = record_pipeline_report(report, component="ingestion", tracker=tracker, stage="run_ingestion")

    assert len(events) == 1
    assert events[0].error_code == ErrorCode.INGEST_SCHEMA_WARNING.value
    assert events[0].severity == "WARNING"


def test_summarize_events(tmp_path: Path):
    tracker = ErrorTracker(log_dir=tmp_path)
    report = {"status": "failed", "error": "Gold row count mismatch after cost join"}
    events = record_pipeline_report(report, component="gold", tracker=tracker, stage="run_gold")
    summary = summarize_error_events(events)

    assert summary["errors_recorded"] == 1
    assert ErrorCode.GOLD_ROW_COUNT_MISMATCH.value in summary["codes"]
