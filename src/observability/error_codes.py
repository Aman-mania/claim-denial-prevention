"""
Central error-code catalog for the Claim Denial Prevention System.

Error-code format: <DOMAIN>_<NNN>. Treat ErrorCode values as public operational
contracts; dashboards, logs, tests, and future CloudWatch/Databricks monitoring
should key off these stable codes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ErrorSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ErrorCategory(str, Enum):
    CONFIGURATION = "configuration"
    DATA_QUALITY = "data_quality"
    FILE_IO = "file_io"
    SCHEMA_VALIDATION = "schema_validation"
    PIPELINE_EXECUTION = "pipeline_execution"
    MODEL_TRAINING = "model_training"
    MODEL_INFERENCE = "model_inference"
    EXPLAINABILITY = "explainability"
    RAG = "rag"
    OBSERVABILITY = "observability"
    UNKNOWN = "unknown"


class ErrorCode(str, Enum):
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

    # Week 5 explainability / XAI
    XAI_GOLD_FEATURES_MISSING = "XAI_001"
    XAI_SHAP_EXPLAINER_MISSING = "XAI_002"
    XAI_MODEL_EXPLAINER_MISMATCH = "XAI_003"
    XAI_REASON_MAPPING_FAILED = "XAI_004"
    XAI_EXPLANATION_GENERATION_FAILED = "XAI_005"
    XAI_EXPLANATION_PARTIAL_FAILURE = "XAI_006"
    XAI_EXPLANATION_WRITE_FAILED = "XAI_007"
    XAI_NO_REASON_GENERATED = "XAI_008"
    XAI_FEATURE_BUILD_FAILED = "XAI_009"
    XAI_UNEXPECTED = "XAI_999"

    # Week 6 RAG placeholders. These are added now so Week 6 can reuse the same
    # error-handling contract without another large observability refactor.
    RAG_POLICY_DOCUMENT_MISSING = "RAG_001"
    RAG_DOCUMENT_PARSE_FAILED = "RAG_002"
    RAG_EMBEDDING_MODEL_LOAD_FAILED = "RAG_003"
    RAG_EMBEDDING_GENERATION_FAILED = "RAG_004"
    RAG_VECTOR_INDEX_MISSING = "RAG_005"
    RAG_RETRIEVAL_FAILED = "RAG_006"
    RAG_NO_RELEVANT_POLICY_FOUND = "RAG_007"
    RAG_UNEXPECTED = "RAG_999"

    # Configuration / system
    CONFIG_INVALID = "CONFIG_001"
    OBS_ERROR_TRACKER_FAILED = "OBS_001"
    SYSTEM_UNEXPECTED = "SYSTEM_999"


@dataclass(frozen=True)
class ErrorDefinition:
    code: ErrorCode
    name: str
    category: ErrorCategory
    default_severity: ErrorSeverity
    retryable: bool
    user_message: str


def _def(
    code: ErrorCode,
    name: str,
    category: ErrorCategory,
    severity: ErrorSeverity,
    retryable: bool,
    message: str,
) -> ErrorDefinition:
    return ErrorDefinition(code, name, category, severity, retryable, message)


ERROR_DEFINITIONS: dict[ErrorCode, ErrorDefinition] = {
    # Ingestion
    ErrorCode.INGEST_RAW_FILE_MISSING: _def(ErrorCode.INGEST_RAW_FILE_MISSING, "Raw input file missing", ErrorCategory.FILE_IO, ErrorSeverity.ERROR, True, "A required raw CSV file is missing. Place it in data/raw and rerun ingestion."),
    ErrorCode.INGEST_SCHEMA_WARNING: _def(ErrorCode.INGEST_SCHEMA_WARNING, "Bronze schema validation warning", ErrorCategory.SCHEMA_VALIDATION, ErrorSeverity.WARNING, False, "Raw data did not fully match the expected schema. Bronze preserved the data for Silver review."),
    ErrorCode.INGEST_SCHEMA_ERROR: _def(ErrorCode.INGEST_SCHEMA_ERROR, "Bronze schema validation error", ErrorCategory.SCHEMA_VALIDATION, ErrorSeverity.ERROR, False, "Schema validation failed unexpectedly during ingestion."),
    ErrorCode.INGEST_WRITE_FAILED: _def(ErrorCode.INGEST_WRITE_FAILED, "Bronze write failed", ErrorCategory.FILE_IO, ErrorSeverity.ERROR, True, "The Bronze Parquet file could not be written."),
    ErrorCode.INGEST_UNKNOWN_DATASET: _def(ErrorCode.INGEST_UNKNOWN_DATASET, "Unknown ingestion dataset", ErrorCategory.CONFIGURATION, ErrorSeverity.ERROR, False, "The requested dataset is not registered in the ingestion registry."),

    # Silver
    ErrorCode.SILVER_BRONZE_FILE_MISSING: _def(ErrorCode.SILVER_BRONZE_FILE_MISSING, "Bronze file missing for Silver", ErrorCategory.FILE_IO, ErrorSeverity.ERROR, True, "A Bronze input file is missing. Run ingestion before Silver cleaning."),
    ErrorCode.SILVER_VALIDATION_WARNING: _def(ErrorCode.SILVER_VALIDATION_WARNING, "Silver validation warning", ErrorCategory.SCHEMA_VALIDATION, ErrorSeverity.WARNING, False, "Silver data did not fully match schema expectations. Review the validation report."),
    ErrorCode.SILVER_VALIDATION_ERROR: _def(ErrorCode.SILVER_VALIDATION_ERROR, "Silver validation error", ErrorCategory.SCHEMA_VALIDATION, ErrorSeverity.ERROR, False, "Silver validation failed unexpectedly."),
    ErrorCode.SILVER_NEGATIVE_AMOUNT_NULLIFIED: _def(ErrorCode.SILVER_NEGATIVE_AMOUNT_NULLIFIED, "Negative amount nullified", ErrorCategory.DATA_QUALITY, ErrorSeverity.WARNING, False, "One or more negative billed amounts were converted to null."),
    ErrorCode.SILVER_DUPLICATES_REMOVED: _def(ErrorCode.SILVER_DUPLICATES_REMOVED, "Duplicates removed in Silver", ErrorCategory.DATA_QUALITY, ErrorSeverity.WARNING, False, "Duplicate IDs were detected and only the first record was kept."),
    ErrorCode.SILVER_WRITE_FAILED: _def(ErrorCode.SILVER_WRITE_FAILED, "Silver write failed", ErrorCategory.FILE_IO, ErrorSeverity.ERROR, True, "The Silver Parquet file could not be written."),

    # Gold
    ErrorCode.GOLD_SILVER_FILE_MISSING: _def(ErrorCode.GOLD_SILVER_FILE_MISSING, "Silver file missing for Gold", ErrorCategory.FILE_IO, ErrorSeverity.ERROR, True, "A Silver input file is missing. Run Silver cleaning before Gold feature engineering."),
    ErrorCode.GOLD_LABEL_MISSING: _def(ErrorCode.GOLD_LABEL_MISSING, "Denial label missing", ErrorCategory.DATA_QUALITY, ErrorSeverity.ERROR, False, "No usable denial_flag was found and synthetic-label fallback was unavailable."),
    ErrorCode.GOLD_ROW_COUNT_MISMATCH: _def(ErrorCode.GOLD_ROW_COUNT_MISMATCH, "Gold row count mismatch", ErrorCategory.DATA_QUALITY, ErrorSeverity.CRITICAL, False, "Gold feature engineering changed the claim row count. Check joins for duplication or row loss."),
    ErrorCode.GOLD_COST_JOIN_DUPLICATED_ROWS: _def(ErrorCode.GOLD_COST_JOIN_DUPLICATED_ROWS, "Cost join duplicated rows", ErrorCategory.DATA_QUALITY, ErrorSeverity.CRITICAL, False, "The cost join created duplicate claim rows. Use regional match plus procedure-level fallback."),
    ErrorCode.GOLD_FEATURE_MISSING: _def(ErrorCode.GOLD_FEATURE_MISSING, "Gold feature missing", ErrorCategory.DATA_QUALITY, ErrorSeverity.ERROR, False, "An expected ML feature was not produced by the Gold pipeline."),
    ErrorCode.GOLD_ARTIFACT_WRITE_FAILED: _def(ErrorCode.GOLD_ARTIFACT_WRITE_FAILED, "Gold artifact write failed", ErrorCategory.FILE_IO, ErrorSeverity.ERROR, True, "A Gold artifact could not be written."),

    # ML
    ErrorCode.ML_GOLD_FEATURES_MISSING: _def(ErrorCode.ML_GOLD_FEATURES_MISSING, "Gold feature table missing", ErrorCategory.FILE_IO, ErrorSeverity.ERROR, True, "The Gold feature table is missing. Run Gold before training."),
    ErrorCode.ML_TARGET_MISSING: _def(ErrorCode.ML_TARGET_MISSING, "Training target missing", ErrorCategory.DATA_QUALITY, ErrorSeverity.ERROR, False, "The model target column denial_flag is missing or invalid."),
    ErrorCode.ML_TRAIN_SPLIT_FAILED: _def(ErrorCode.ML_TRAIN_SPLIT_FAILED, "Train/validation/test split failed", ErrorCategory.MODEL_TRAINING, ErrorSeverity.ERROR, False, "The dataset could not be split for training. Check row count and class balance."),
    ErrorCode.ML_TRAINING_FAILED: _def(ErrorCode.ML_TRAINING_FAILED, "Model training failed", ErrorCategory.MODEL_TRAINING, ErrorSeverity.ERROR, False, "Model training failed. Review feature data and dependency versions."),
    ErrorCode.ML_THRESHOLD_TUNING_FAILED: _def(ErrorCode.ML_THRESHOLD_TUNING_FAILED, "Threshold tuning failed", ErrorCategory.MODEL_TRAINING, ErrorSeverity.ERROR, False, "Risk-threshold tuning failed. Check validation predictions and labels."),
    ErrorCode.ML_CALIBRATION_FAILED: _def(ErrorCode.ML_CALIBRATION_FAILED, "Calibration report failed", ErrorCategory.MODEL_TRAINING, ErrorSeverity.WARNING, False, "Calibration report generation failed. Probability quality is unknown."),
    ErrorCode.ML_MODEL_SAVE_FAILED: _def(ErrorCode.ML_MODEL_SAVE_FAILED, "Model artifact save failed", ErrorCategory.FILE_IO, ErrorSeverity.ERROR, True, "A model artifact could not be saved."),

    # Inference
    ErrorCode.INFER_MODEL_NOT_FOUND: _def(ErrorCode.INFER_MODEL_NOT_FOUND, "Inference model not found", ErrorCategory.FILE_IO, ErrorSeverity.ERROR, True, "No trained model was found. Run training before inference."),
    ErrorCode.INFER_ARTIFACT_NOT_FOUND: _def(ErrorCode.INFER_ARTIFACT_NOT_FOUND, "Inference artifact not found", ErrorCategory.FILE_IO, ErrorSeverity.ERROR, True, "Feature lookup artifacts are missing. Run Gold before inference."),
    ErrorCode.INFER_INVALID_CLAIM: _def(ErrorCode.INFER_INVALID_CLAIM, "Invalid custom claim", ErrorCategory.DATA_QUALITY, ErrorSeverity.ERROR, False, "The custom claim input is invalid. Correct the fields and retry."),
    ErrorCode.INFER_FEATURE_BUILD_FAILED: _def(ErrorCode.INFER_FEATURE_BUILD_FAILED, "Feature build failed during inference", ErrorCategory.MODEL_INFERENCE, ErrorSeverity.ERROR, False, "The claim could not be converted into model features."),
    ErrorCode.INFER_PREDICTION_FAILED: _def(ErrorCode.INFER_PREDICTION_FAILED, "Prediction failed", ErrorCategory.MODEL_INFERENCE, ErrorSeverity.ERROR, False, "The model failed to produce a prediction."),
    ErrorCode.INFER_INCONSISTENT_CLAIM_STATE: _def(ErrorCode.INFER_INCONSISTENT_CLAIM_STATE, "Inconsistent custom claim state", ErrorCategory.DATA_QUALITY, ErrorSeverity.WARNING, False, "The custom claim contains contradictory fields."),

    # XAI / Explainability
    ErrorCode.XAI_GOLD_FEATURES_MISSING: _def(ErrorCode.XAI_GOLD_FEATURES_MISSING, "Gold features missing for explainability", ErrorCategory.FILE_IO, ErrorSeverity.ERROR, True, "Gold claim features are missing. Run run_gold.py before run_explain.py."),
    ErrorCode.XAI_SHAP_EXPLAINER_MISSING: _def(ErrorCode.XAI_SHAP_EXPLAINER_MISSING, "SHAP explainer model missing", ErrorCategory.EXPLAINABILITY, ErrorSeverity.ERROR, True, "The XGBoost model required for SHAP explanations is missing. Run run_train.py."),
    ErrorCode.XAI_MODEL_EXPLAINER_MISMATCH: _def(ErrorCode.XAI_MODEL_EXPLAINER_MISMATCH, "Prediction/explanation model mismatch", ErrorCategory.EXPLAINABILITY, ErrorSeverity.ERROR, False, "Prediction and explanation models do not match. Use XGBoost for Week 5 explanations."),
    ErrorCode.XAI_REASON_MAPPING_FAILED: _def(ErrorCode.XAI_REASON_MAPPING_FAILED, "Reason mapping failed", ErrorCategory.EXPLAINABILITY, ErrorSeverity.ERROR, False, "A SHAP feature could not be mapped to a business reason."),
    ErrorCode.XAI_EXPLANATION_GENERATION_FAILED: _def(ErrorCode.XAI_EXPLANATION_GENERATION_FAILED, "Explanation generation failed", ErrorCategory.EXPLAINABILITY, ErrorSeverity.ERROR, False, "The system failed to generate an explanation for a claim."),
    ErrorCode.XAI_EXPLANATION_PARTIAL_FAILURE: _def(ErrorCode.XAI_EXPLANATION_PARTIAL_FAILURE, "Partial explanation generation failure", ErrorCategory.EXPLAINABILITY, ErrorSeverity.WARNING, False, "Some claims failed during explanation generation. Review failed_claims in the report."),
    ErrorCode.XAI_EXPLANATION_WRITE_FAILED: _def(ErrorCode.XAI_EXPLANATION_WRITE_FAILED, "Explanation table write failed", ErrorCategory.FILE_IO, ErrorSeverity.ERROR, True, "The explanation output table could not be written."),
    ErrorCode.XAI_NO_REASON_GENERATED: _def(ErrorCode.XAI_NO_REASON_GENERATED, "No business reason generated", ErrorCategory.EXPLAINABILITY, ErrorSeverity.WARNING, False, "No mapped business reason was generated for the claim."),
    ErrorCode.XAI_FEATURE_BUILD_FAILED: _def(ErrorCode.XAI_FEATURE_BUILD_FAILED, "XAI feature build failed", ErrorCategory.EXPLAINABILITY, ErrorSeverity.ERROR, False, "The raw claim could not be converted into features for explanation."),

    # RAG placeholders
    ErrorCode.RAG_POLICY_DOCUMENT_MISSING: _def(ErrorCode.RAG_POLICY_DOCUMENT_MISSING, "Policy document missing", ErrorCategory.RAG, ErrorSeverity.ERROR, True, "A policy document is missing from the RAG corpus."),
    ErrorCode.RAG_DOCUMENT_PARSE_FAILED: _def(ErrorCode.RAG_DOCUMENT_PARSE_FAILED, "Policy document parse failed", ErrorCategory.RAG, ErrorSeverity.ERROR, False, "A policy document could not be parsed."),
    ErrorCode.RAG_EMBEDDING_MODEL_LOAD_FAILED: _def(ErrorCode.RAG_EMBEDDING_MODEL_LOAD_FAILED, "Embedding model load failed", ErrorCategory.RAG, ErrorSeverity.ERROR, True, "The embedding model could not be loaded."),
    ErrorCode.RAG_EMBEDDING_GENERATION_FAILED: _def(ErrorCode.RAG_EMBEDDING_GENERATION_FAILED, "Embedding generation failed", ErrorCategory.RAG, ErrorSeverity.ERROR, False, "Text embeddings could not be generated."),
    ErrorCode.RAG_VECTOR_INDEX_MISSING: _def(ErrorCode.RAG_VECTOR_INDEX_MISSING, "Vector index missing", ErrorCategory.RAG, ErrorSeverity.ERROR, True, "The vector index is missing. Run policy ingestion before retrieval."),
    ErrorCode.RAG_RETRIEVAL_FAILED: _def(ErrorCode.RAG_RETRIEVAL_FAILED, "Policy retrieval failed", ErrorCategory.RAG, ErrorSeverity.ERROR, False, "Policy retrieval failed for a reason query."),
    ErrorCode.RAG_NO_RELEVANT_POLICY_FOUND: _def(ErrorCode.RAG_NO_RELEVANT_POLICY_FOUND, "No relevant policy found", ErrorCategory.RAG, ErrorSeverity.WARNING, False, "No relevant policy chunk met the retrieval threshold."),

    # Config/system
    ErrorCode.CONFIG_INVALID: _def(ErrorCode.CONFIG_INVALID, "Invalid configuration", ErrorCategory.CONFIGURATION, ErrorSeverity.ERROR, False, "Configuration is invalid or incomplete."),
    ErrorCode.OBS_ERROR_TRACKER_FAILED: _def(ErrorCode.OBS_ERROR_TRACKER_FAILED, "Error tracker failed", ErrorCategory.OBSERVABILITY, ErrorSeverity.WARNING, True, "The error tracker failed to persist an event, but execution continued."),
}


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
