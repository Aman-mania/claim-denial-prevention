"""
Central error-code catalog for the Claim Denial Prevention System.

Why this exists
---------------
- Every operational failure gets a stable, searchable code.
- Error codes are constants, not scattered string literals.
- Local JSONL logs can be migrated to CloudWatch Logs / Databricks logs later
  without changing call sites.

Error-code format
-----------------
<DOMAIN>_<NNN>

Examples:
- INGEST_001: raw input file is missing
- GOLD_003: Gold row count no longer matches claims row count
- INFER_004: model prediction failed
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ErrorSeverity(str, Enum):
    """Operational severity used for logs, reports, and future alert routing."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ErrorCategory(str, Enum):
    """Broad category for dashboard filtering and CloudWatch/Databricks metrics."""

    CONFIGURATION = "configuration"
    DATA_QUALITY = "data_quality"
    FILE_IO = "file_io"
    SCHEMA_VALIDATION = "schema_validation"
    PIPELINE_EXECUTION = "pipeline_execution"
    MODEL_TRAINING = "model_training"
    MODEL_INFERENCE = "model_inference"
    OBSERVABILITY = "observability"
    UNKNOWN = "unknown"


class ErrorCode(str, Enum):
    """Stable application error codes. Treat these as public API."""

    # Ingestion / Bronze
    INGEST_RAW_FILE_MISSING = "INGEST_001"
    INGEST_SCHEMA_WARNING = "INGEST_002"
    INGEST_SCHEMA_ERROR = "INGEST_003"
    INGEST_WRITE_FAILED = "INGEST_004"
    INGEST_UNKNOWN_DATASET = "INGEST_005"
    INGEST_UNEXPECTED = "INGEST_999"

    # Silver
    SILVER_BRONZE_FILE_MISSING = "SILVER_001"
    SILVER_VALIDATION_WARNING = "SILVER_002"
    SILVER_VALIDATION_ERROR = "SILVER_003"
    SILVER_NEGATIVE_AMOUNT_NULLIFIED = "SILVER_004"
    SILVER_DUPLICATES_REMOVED = "SILVER_005"
    SILVER_WRITE_FAILED = "SILVER_006"
    SILVER_UNEXPECTED = "SILVER_999"

    # Gold / features
    GOLD_SILVER_FILE_MISSING = "GOLD_001"
    GOLD_LABEL_MISSING = "GOLD_002"
    GOLD_ROW_COUNT_MISMATCH = "GOLD_003"
    GOLD_COST_JOIN_DUPLICATED_ROWS = "GOLD_004"
    GOLD_FEATURE_MISSING = "GOLD_005"
    GOLD_ARTIFACT_WRITE_FAILED = "GOLD_006"
    GOLD_UNEXPECTED = "GOLD_999"

    # ML training
    ML_GOLD_FEATURES_MISSING = "ML_001"
    ML_TARGET_MISSING = "ML_002"
    ML_TRAIN_SPLIT_FAILED = "ML_003"
    ML_TRAINING_FAILED = "ML_004"
    ML_THRESHOLD_TUNING_FAILED = "ML_005"
    ML_CALIBRATION_FAILED = "ML_006"
    ML_MODEL_SAVE_FAILED = "ML_007"
    ML_UNEXPECTED = "ML_999"

    # Inference / custom claim builder / API
    INFER_MODEL_NOT_FOUND = "INFER_001"
    INFER_ARTIFACT_NOT_FOUND = "INFER_002"
    INFER_INVALID_CLAIM = "INFER_003"
    INFER_FEATURE_BUILD_FAILED = "INFER_004"
    INFER_PREDICTION_FAILED = "INFER_005"
    INFER_INCONSISTENT_CLAIM_STATE = "INFER_006"
    INFER_UNEXPECTED = "INFER_999"

    # Configuration / system
    CONFIG_INVALID = "CONFIG_001"
    OBS_ERROR_TRACKER_FAILED = "OBS_001"
    SYSTEM_UNEXPECTED = "SYSTEM_999"


@dataclass(frozen=True)
class ErrorDefinition:
    """Metadata describing an error code."""

    code: ErrorCode
    name: str
    category: ErrorCategory
    default_severity: ErrorSeverity
    retryable: bool
    user_message: str


ERROR_DEFINITIONS: dict[ErrorCode, ErrorDefinition] = {
    ErrorCode.INGEST_RAW_FILE_MISSING: ErrorDefinition(
        ErrorCode.INGEST_RAW_FILE_MISSING,
        "Raw input file missing",
        ErrorCategory.FILE_IO,
        ErrorSeverity.ERROR,
        retryable=True,
        user_message="A required raw CSV file is missing. Place the file in data/raw and rerun ingestion.",
    ),
    ErrorCode.INGEST_SCHEMA_WARNING: ErrorDefinition(
        ErrorCode.INGEST_SCHEMA_WARNING,
        "Bronze schema validation warning",
        ErrorCategory.SCHEMA_VALIDATION,
        ErrorSeverity.WARNING,
        retryable=False,
        user_message="Raw data did not fully match the expected schema. Bronze preserved the data for Silver review.",
    ),
    ErrorCode.INGEST_SCHEMA_ERROR: ErrorDefinition(
        ErrorCode.INGEST_SCHEMA_ERROR,
        "Bronze schema validation error",
        ErrorCategory.SCHEMA_VALIDATION,
        ErrorSeverity.ERROR,
        retryable=False,
        user_message="Schema validation failed unexpectedly during ingestion.",
    ),
    ErrorCode.INGEST_WRITE_FAILED: ErrorDefinition(
        ErrorCode.INGEST_WRITE_FAILED,
        "Bronze write failed",
        ErrorCategory.FILE_IO,
        ErrorSeverity.ERROR,
        retryable=True,
        user_message="The Bronze Parquet file could not be written.",
    ),
    ErrorCode.INGEST_UNKNOWN_DATASET: ErrorDefinition(
        ErrorCode.INGEST_UNKNOWN_DATASET,
        "Unknown ingestion dataset",
        ErrorCategory.CONFIGURATION,
        ErrorSeverity.ERROR,
        retryable=False,
        user_message="The requested dataset is not registered in the ingestion registry.",
    ),
    ErrorCode.SILVER_BRONZE_FILE_MISSING: ErrorDefinition(
        ErrorCode.SILVER_BRONZE_FILE_MISSING,
        "Bronze file missing for Silver",
        ErrorCategory.FILE_IO,
        ErrorSeverity.ERROR,
        retryable=True,
        user_message="A Bronze input file is missing. Run ingestion before Silver cleaning.",
    ),
    ErrorCode.SILVER_VALIDATION_WARNING: ErrorDefinition(
        ErrorCode.SILVER_VALIDATION_WARNING,
        "Silver validation warning",
        ErrorCategory.SCHEMA_VALIDATION,
        ErrorSeverity.WARNING,
        retryable=False,
        user_message="Silver data did not fully match schema expectations. Review the validation report.",
    ),
    ErrorCode.SILVER_VALIDATION_ERROR: ErrorDefinition(
        ErrorCode.SILVER_VALIDATION_ERROR,
        "Silver validation error",
        ErrorCategory.SCHEMA_VALIDATION,
        ErrorSeverity.ERROR,
        retryable=False,
        user_message="Silver validation failed unexpectedly.",
    ),
    ErrorCode.SILVER_NEGATIVE_AMOUNT_NULLIFIED: ErrorDefinition(
        ErrorCode.SILVER_NEGATIVE_AMOUNT_NULLIFIED,
        "Negative amount nullified",
        ErrorCategory.DATA_QUALITY,
        ErrorSeverity.WARNING,
        retryable=False,
        user_message="One or more negative billed amounts were invalid and were converted to null.",
    ),
    ErrorCode.SILVER_DUPLICATES_REMOVED: ErrorDefinition(
        ErrorCode.SILVER_DUPLICATES_REMOVED,
        "Duplicates removed in Silver",
        ErrorCategory.DATA_QUALITY,
        ErrorSeverity.WARNING,
        retryable=False,
        user_message="Duplicate IDs were detected and only the first record was kept.",
    ),
    ErrorCode.SILVER_WRITE_FAILED: ErrorDefinition(
        ErrorCode.SILVER_WRITE_FAILED,
        "Silver write failed",
        ErrorCategory.FILE_IO,
        ErrorSeverity.ERROR,
        retryable=True,
        user_message="The Silver Parquet file could not be written.",
    ),
    ErrorCode.GOLD_SILVER_FILE_MISSING: ErrorDefinition(
        ErrorCode.GOLD_SILVER_FILE_MISSING,
        "Silver file missing for Gold",
        ErrorCategory.FILE_IO,
        ErrorSeverity.ERROR,
        retryable=True,
        user_message="A Silver input file is missing. Run Silver cleaning before Gold feature engineering.",
    ),
    ErrorCode.GOLD_LABEL_MISSING: ErrorDefinition(
        ErrorCode.GOLD_LABEL_MISSING,
        "Denial label missing",
        ErrorCategory.DATA_QUALITY,
        ErrorSeverity.ERROR,
        retryable=False,
        user_message="No usable denial_flag was found and synthetic-label fallback was unavailable.",
    ),
    ErrorCode.GOLD_ROW_COUNT_MISMATCH: ErrorDefinition(
        ErrorCode.GOLD_ROW_COUNT_MISMATCH,
        "Gold row count mismatch",
        ErrorCategory.DATA_QUALITY,
        ErrorSeverity.CRITICAL,
        retryable=False,
        user_message="Gold feature engineering changed the claim row count. Check joins for duplication or row loss.",
    ),
    ErrorCode.GOLD_COST_JOIN_DUPLICATED_ROWS: ErrorDefinition(
        ErrorCode.GOLD_COST_JOIN_DUPLICATED_ROWS,
        "Cost join duplicated rows",
        ErrorCategory.DATA_QUALITY,
        ErrorSeverity.CRITICAL,
        retryable=False,
        user_message="The cost join created duplicate claim rows. Use regional match plus procedure-level fallback.",
    ),
    ErrorCode.GOLD_FEATURE_MISSING: ErrorDefinition(
        ErrorCode.GOLD_FEATURE_MISSING,
        "Gold feature missing",
        ErrorCategory.DATA_QUALITY,
        ErrorSeverity.ERROR,
        retryable=False,
        user_message="An expected ML feature was not produced by the Gold pipeline.",
    ),
    ErrorCode.GOLD_ARTIFACT_WRITE_FAILED: ErrorDefinition(
        ErrorCode.GOLD_ARTIFACT_WRITE_FAILED,
        "Gold artifact write failed",
        ErrorCategory.FILE_IO,
        ErrorSeverity.ERROR,
        retryable=True,
        user_message="A Gold artifact could not be written.",
    ),
    ErrorCode.ML_GOLD_FEATURES_MISSING: ErrorDefinition(
        ErrorCode.ML_GOLD_FEATURES_MISSING,
        "Gold feature table missing",
        ErrorCategory.FILE_IO,
        ErrorSeverity.ERROR,
        retryable=True,
        user_message="The Gold feature table is missing. Run Gold before training.",
    ),
    ErrorCode.ML_TARGET_MISSING: ErrorDefinition(
        ErrorCode.ML_TARGET_MISSING,
        "Training target missing",
        ErrorCategory.DATA_QUALITY,
        ErrorSeverity.ERROR,
        retryable=False,
        user_message="The model target column denial_flag is missing or invalid.",
    ),
    ErrorCode.ML_TRAIN_SPLIT_FAILED: ErrorDefinition(
        ErrorCode.ML_TRAIN_SPLIT_FAILED,
        "Train/validation/test split failed",
        ErrorCategory.MODEL_TRAINING,
        ErrorSeverity.ERROR,
        retryable=False,
        user_message="The dataset could not be split for training. Check row count and class balance.",
    ),
    ErrorCode.ML_TRAINING_FAILED: ErrorDefinition(
        ErrorCode.ML_TRAINING_FAILED,
        "Model training failed",
        ErrorCategory.MODEL_TRAINING,
        ErrorSeverity.ERROR,
        retryable=False,
        user_message="Model training failed. Review feature data and dependency versions.",
    ),
    ErrorCode.ML_THRESHOLD_TUNING_FAILED: ErrorDefinition(
        ErrorCode.ML_THRESHOLD_TUNING_FAILED,
        "Threshold tuning failed",
        ErrorCategory.MODEL_TRAINING,
        ErrorSeverity.ERROR,
        retryable=False,
        user_message="Risk-threshold tuning failed. Check validation predictions and labels.",
    ),
    ErrorCode.ML_CALIBRATION_FAILED: ErrorDefinition(
        ErrorCode.ML_CALIBRATION_FAILED,
        "Calibration report failed",
        ErrorCategory.MODEL_TRAINING,
        ErrorSeverity.WARNING,
        retryable=False,
        user_message="Calibration report generation failed. The model may still be usable, but probability quality is unknown.",
    ),
    ErrorCode.ML_MODEL_SAVE_FAILED: ErrorDefinition(
        ErrorCode.ML_MODEL_SAVE_FAILED,
        "Model artifact save failed",
        ErrorCategory.FILE_IO,
        ErrorSeverity.ERROR,
        retryable=True,
        user_message="A model artifact could not be saved.",
    ),
    ErrorCode.INFER_MODEL_NOT_FOUND: ErrorDefinition(
        ErrorCode.INFER_MODEL_NOT_FOUND,
        "Inference model not found",
        ErrorCategory.FILE_IO,
        ErrorSeverity.ERROR,
        retryable=True,
        user_message="No trained model was found. Run training before inference.",
    ),
    ErrorCode.INFER_ARTIFACT_NOT_FOUND: ErrorDefinition(
        ErrorCode.INFER_ARTIFACT_NOT_FOUND,
        "Inference artifact not found",
        ErrorCategory.FILE_IO,
        ErrorSeverity.ERROR,
        retryable=True,
        user_message="Feature lookup artifacts are missing. Run Gold before inference.",
    ),
    ErrorCode.INFER_INVALID_CLAIM: ErrorDefinition(
        ErrorCode.INFER_INVALID_CLAIM,
        "Invalid custom claim",
        ErrorCategory.DATA_QUALITY,
        ErrorSeverity.ERROR,
        retryable=False,
        user_message="The custom claim input is invalid. Correct the fields and retry.",
    ),
    ErrorCode.INFER_FEATURE_BUILD_FAILED: ErrorDefinition(
        ErrorCode.INFER_FEATURE_BUILD_FAILED,
        "Feature build failed during inference",
        ErrorCategory.MODEL_INFERENCE,
        ErrorSeverity.ERROR,
        retryable=False,
        user_message="The claim could not be converted into model features.",
    ),
    ErrorCode.INFER_PREDICTION_FAILED: ErrorDefinition(
        ErrorCode.INFER_PREDICTION_FAILED,
        "Prediction failed",
        ErrorCategory.MODEL_INFERENCE,
        ErrorSeverity.ERROR,
        retryable=False,
        user_message="The model failed to produce a prediction.",
    ),
    ErrorCode.INFER_INCONSISTENT_CLAIM_STATE: ErrorDefinition(
        ErrorCode.INFER_INCONSISTENT_CLAIM_STATE,
        "Inconsistent custom claim state",
        ErrorCategory.DATA_QUALITY,
        ErrorSeverity.WARNING,
        retryable=False,
        user_message="The custom claim contains contradictory fields, such as amount present and amount_missing=true.",
    ),
    ErrorCode.CONFIG_INVALID: ErrorDefinition(
        ErrorCode.CONFIG_INVALID,
        "Invalid configuration",
        ErrorCategory.CONFIGURATION,
        ErrorSeverity.ERROR,
        retryable=False,
        user_message="Configuration is invalid or incomplete.",
    ),
    ErrorCode.OBS_ERROR_TRACKER_FAILED: ErrorDefinition(
        ErrorCode.OBS_ERROR_TRACKER_FAILED,
        "Error tracker failed",
        ErrorCategory.OBSERVABILITY,
        ErrorSeverity.WARNING,
        retryable=True,
        user_message="The error tracker failed to persist an event, but pipeline execution continued.",
    ),
}

# Generic fallback definitions are generated dynamically for *_999 codes.


def get_error_definition(code: ErrorCode | str) -> ErrorDefinition:
    """Return metadata for a code, falling back to a generic definition."""
    error_code = code if isinstance(code, ErrorCode) else ErrorCode(code)
    if error_code in ERROR_DEFINITIONS:
        return ERROR_DEFINITIONS[error_code]

    return ErrorDefinition(
        code=error_code,
        name="Unexpected error",
        category=ErrorCategory.UNKNOWN,
        default_severity=ErrorSeverity.ERROR,
        retryable=False,
        user_message="An unexpected error occurred. Check logs for details.",
    )
