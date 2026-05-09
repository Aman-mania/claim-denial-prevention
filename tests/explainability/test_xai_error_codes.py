import pytest

from src.observability import ErrorCode
from src.observability.error_codes import get_error_definition, ErrorCategory

pytestmark = [pytest.mark.unit, pytest.mark.week5]


def test_xai_error_codes_are_registered():
    definition = get_error_definition(ErrorCode.XAI_EXPLANATION_GENERATION_FAILED)
    assert definition.category == ErrorCategory.EXPLAINABILITY
    assert definition.user_message


def test_rag_placeholder_error_codes_are_ready_for_week6():
    definition = get_error_definition(ErrorCode.RAG_RETRIEVAL_FAILED)
    assert definition.category == ErrorCategory.RAG
