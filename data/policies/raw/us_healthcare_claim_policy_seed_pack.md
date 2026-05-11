---
document_id: us_healthcare_claim_policy_seed_pack
source_type: curated_policy_seed
publisher: Claim Denial Prevention Project
tags: [claim_completeness, medical_necessity, ncci, coding, documentation, prior_authorization, cost, internal]
version: week6_policy_corpus_v1
---

# US Healthcare Claim Denial Policy Seed Pack

This curated seed pack gives the Week 6 RAG layer reliable policy-like passages that align with the project’s current reason codes. It is not legal advice, payer policy, or a substitute for official CMS or payer documents. It summarizes common claim-validation principles and points to source categories in `policy_docs/official_policy_source_registry.json`.

## Claim completeness: diagnosis code required

A claim should include a valid diagnosis code when the diagnosis is needed to support the billed service. Missing diagnosis information can prevent a payer from verifying medical necessity and may result in rejection, denial, or manual review. For RAG mapping, this passage supports reasons such as `MISSING_DIAGNOSIS`, `PROCEDURE_WITHOUT_DIAGNOSIS`, and `CLAIM_COMPLETENESS_RISK`.

Suggested fix: add the correct diagnosis code and ensure it supports the procedure or service billed.

Policy tags: diagnosis, medical_necessity, claim_completeness, documentation

## Claim completeness: procedure code required

A claim should include the procedure, service, supply, or HCPCS/CPT-style code that describes what was performed or provided. Missing procedure information can prevent payment adjudication, fee calculation, and coding-policy checks. For RAG mapping, this passage supports reasons such as `MISSING_PROCEDURE` and `DIAGNOSIS_WITHOUT_PROCEDURE`.

Suggested fix: add the appropriate procedure/service code and verify it matches the clinical documentation.

Policy tags: procedure, cpt, hcpcs, claim_completeness, coding

## Claim completeness: billed amount required

A submitted claim should include the billed amount or charge for the billed line/service when payment calculation depends on the claim charge. Missing billed amount information can cause incomplete claim handling or require manual correction. For RAG mapping, this passage supports `MISSING_AMOUNT`.

Suggested fix: enter the billed amount and validate it against expected contractual or benchmark costs.

Policy tags: billed_amount, charge, claim_completeness, payment

## Medical necessity support

A service generally needs documentation that supports why it was reasonable, necessary, clinically appropriate, and connected to the patient’s diagnosis or condition. If a procedure is present without diagnosis support, or if documentation is absent for a high-cost service, the claim should be flagged for review before submission.

Suggested fix: attach or verify clinical documentation, diagnosis linkage, medical necessity notes, and any payer-required supporting records.

Policy tags: medical_necessity, diagnosis, documentation, procedure, clinical_support

## Correct coding and incompatible procedure combinations

Correct coding policies and edits are used to reduce improper coding and improper payments. Some procedure combinations may be inappropriate when reported together for the same patient, provider, or date of service unless a valid modifier or documented exception applies. This passage supports reasons related to procedure coding, code combinations, and NCCI-style edit risk.

Suggested fix: review the procedure combination, modifier usage, and payer/NCCI edit guidance before submission.

Policy tags: ncci, coding, ptp, mue, modifier, improper_payment

## High-cost claim review

A billed amount that is substantially above the expected benchmark, regional benchmark, or procedure-level average should be reviewed before submission. High-cost claims may require additional documentation, itemization, medical necessity support, prior authorization, or explanation of unusual circumstances.

Suggested fix: verify the billed amount, confirm the expected-cost benchmark, attach supporting documentation, and check whether prior authorization or pre-claim review applies.

Policy tags: high_cost, billed_amount, cost_benchmark, documentation, prior_authorization

## Prior authorization and pre-claim review

Prior authorization and pre-claim review processes are used in some programs to confirm coverage, documentation, and medical-necessity support earlier in the workflow. These processes do not necessarily change medical necessity requirements; they move review earlier to avoid avoidable denials and appeals.

Suggested fix: check payer-specific prior authorization requirements and submit supporting documentation before service or before claim submission when required.

Policy tags: prior_authorization, pre_claim_review, documentation, medical_necessity

## Provider history and repeated quality issues

When a provider has a history of incomplete claims, missing codes, or structural claim-quality issues, new claims from that provider should receive additional validation. Provider history is not itself a denial rule, but it is a useful operational signal for preventing repeated mistakes.

Suggested fix: verify the claim against a provider-specific checklist and review whether recent submissions show repeated missing fields or documentation gaps.

Policy tags: provider_history, quality_review, operational_risk, repeat_error

## Regional or procedure benchmark missing

If no regional or procedure benchmark is available for a procedure, the system should not invent a policy conclusion. It should flag the benchmark gap and rely on available documentation, payer policy, and coding guidance.

Suggested fix: route to manual review or enrich the cost benchmark table before relying on cost-based risk.

Policy tags: cost_benchmark, missing_benchmark, manual_review, data_quality
