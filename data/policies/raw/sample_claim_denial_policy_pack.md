# Sample Claim Denial Prevention Policy Pack

This policy pack is a synthetic educational corpus for local development. It is
not legal, billing, or payer advice. Replace or supplement it with official payer,
CMS, coding, and organizational policy documents before a real deployment.

## Diagnosis Required for Medical Necessity

Claims should include a valid diagnosis code when a procedure or service is
billed. Diagnosis information is used to support medical necessity and payer
adjudication. Missing diagnosis information may prevent a payer from determining
whether the billed service is clinically justified.

If a claim is missing diagnosis information, the billing analyst should add the
appropriate diagnosis code and verify that the diagnosis supports the billed
procedure before submission.

Tags: diagnosis, medical_necessity, claim_completeness

## Procedure Code Required for Payment Review

A claim should include the procedure or service code that describes what was
performed. Missing procedure information may make the claim incomplete because
the payer cannot determine what service is being billed.

If the diagnosis is present but the procedure is missing, the analyst should add
the appropriate procedure code and confirm that the procedure is supported by the
clinical documentation.

Tags: procedure_coding, claim_completeness, documentation

## Procedure Must Be Supported by Diagnosis

When a procedure is billed, the diagnosis should provide clinical context for why
the procedure was medically necessary. A procedure without diagnosis support may
be denied for insufficient medical necessity or incomplete documentation.

The analyst should verify the diagnosis/procedure relationship and attach or
reference supporting documentation when needed.

Tags: diagnosis, procedure_coding, medical_necessity, documentation

## Billed Amount and Cost Justification

Claims with billed amounts significantly above expected cost benchmarks may need
additional review. High-cost claims should be checked for data-entry errors,
incorrect units, duplicate billing, missing modifiers, and supporting medical
justification.

If a billed amount exceeds the expected benchmark, the analyst should verify the
amount, units, and documentation before submission.

Tags: high_cost, documentation, payer_policy

## Missing Billed Amount

A claim should contain the billed amount for the reported service. Missing amount
information makes the claim incomplete and may prevent pricing, adjudication, or
payment review.

If the billed amount is missing, the analyst should add the amount from the bill
or charge capture system and confirm that it matches the procedure and units.

Tags: claim_completeness, high_cost, documentation

## Prior Authorization and High-Value Services

Some high-value, unusual, or payer-sensitive services may require prior
authorization or pre-submission review. When required, authorization information
should be available before claim submission.

The analyst should check payer requirements for services that are high cost,
unusual for the provider specialty, or frequently denied.

Tags: prior_authorization, high_cost, payer_policy, documentation

## Provider Credential and Specialty Review

The rendering or billing provider should be appropriate for the service billed.
If the provider specialty appears inconsistent with the procedure, additional
review may be required.

The analyst should confirm provider credentials, specialty, and service location
when the claim pattern appears unusual.

Tags: provider, procedure_coding, payer_policy

## Duplicate and Frequent Claim Review

Claims that appear duplicated or unusually frequent for the same patient should
be checked before submission. Duplicate submissions or repeated services without
supporting documentation may be denied.

The analyst should verify dates of service, patient identifier, procedure code,
and supporting documentation before submission.

Tags: duplicate, claim_completeness, documentation
