from src.rules.claim_validator import ClaimInputValidator


def test_validator_blocks_missing_required_ids():
    result = ClaimInputValidator().validate({"claim_id": "C1"})
    assert result.is_valid is False
    codes = {issue.code for issue in result.blocking_errors}
    assert "CLAIM_ERR_REQUIRED_FIELD_MISSING" in codes


def test_validator_allows_missing_diagnosis_and_amount_as_warnings():
    result = ClaimInputValidator(known_provider_ids={"PR1"}).validate({
        "claim_id": "C1",
        "patient_id": "P1",
        "provider_id": "PR1",
        "procedure_code": "proc1",
        "billed_amount": "",
    })
    assert result.is_valid is True
    assert result.normalized_claim["procedure_code"] == "PROC1"
    codes = {issue.code for issue in result.warnings}
    assert "CLAIM_WARN_MISSING_DIAGNOSIS" in codes
    assert "CLAIM_WARN_AMOUNT_MISSING" in codes
    assert "CLAIM_WARN_PROC_WITHOUT_DIAGNOSIS" in codes


def test_validator_blocks_invalid_amount():
    result = ClaimInputValidator().validate({
        "claim_id": "C1",
        "patient_id": "P1",
        "provider_id": "PR1",
        "billed_amount": "abc",
    })
    assert result.is_valid is False
    assert any(issue.code == "CLAIM_ERR_INVALID_AMOUNT" for issue in result.blocking_errors)


def test_validator_warns_unknown_reference_values():
    result = ClaimInputValidator(
        known_provider_ids={"PR1"},
        known_diagnosis_codes={"D10"},
        known_procedure_codes={"PROC1"},
    ).validate({
        "claim_id": "C1",
        "patient_id": "P1",
        "provider_id": "PR999",
        "diagnosis_code": "D999",
        "procedure_code": "PROC999",
        "billed_amount": 100,
    })
    assert result.is_valid is True
    codes = {issue.code for issue in result.warnings}
    assert "CLAIM_WARN_UNKNOWN_PROVIDER" in codes
    assert "CLAIM_WARN_UNKNOWN_DIAGNOSIS" in codes
    assert "CLAIM_WARN_UNKNOWN_PROCEDURE" in codes
