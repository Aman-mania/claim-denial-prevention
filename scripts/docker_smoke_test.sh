#!/usr/bin/env bash
set -euo pipefail

API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
UI_BASE_URL="${UI_BASE_URL:-http://localhost:8501}"
ANALYST_EMAIL="${ANALYST_EMAIL:-analyst@example.com}"
ANALYST_PASSWORD="${ANALYST_PASSWORD:-analyst12345}"

printf "Checking API health...\n"
curl -fsS "$API_BASE_URL/health" >/dev/null
printf "API health OK.\n"

printf "Checking UI health...\n"
curl -fsS "$UI_BASE_URL/_stcore/health" >/dev/null
printf "UI health OK.\n"

printf "Logging in...\n"
TOKEN=$(curl -fsS -X POST "$API_BASE_URL/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"$ANALYST_EMAIL\",\"password\":\"$ANALYST_PASSWORD\"}" \
  | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
printf "Login OK.\n"

printf "Validating claim...\n"
curl -fsS -X POST "$API_BASE_URL/claims/validate" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"claim_id":"SMOKE001","patient_id":"P001","provider_id":"PR100","diagnosis_code":"","procedure_code":"PROC1","billed_amount":12000}' >/dev/null
printf "Validation OK.\n"

printf "Requesting recommendation...\n"
curl -fsS -X POST "$API_BASE_URL/claims/recommend" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"claim_id":"SMOKE001","patient_id":"P001","provider_id":"PR100","diagnosis_code":"","procedure_code":"PROC1","billed_amount":12000}' \
  | python -c "import sys,json; data=json.load(sys.stdin); assert data['status']=='success'; print('Recommendation OK:', data['data']['decision']['status'], data['data']['agent_presentation']['source'])"

printf "Smoke test passed.\n"
