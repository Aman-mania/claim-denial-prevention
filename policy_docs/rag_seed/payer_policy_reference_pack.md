---
document_id: payer_policy_reference_pack
source_type: curated_payer_reference
publisher: Claim Denial Prevention Project
tags: [payer, medical_policy, medical_necessity, prior_authorization, documentation, plan_document]
version: week6_policy_corpus_v1
---

# Public Payer Policy Reference Pack

This document summarizes public payer-policy concepts for local RAG testing. It does not copy full payer policy documents and does not replace the payer’s current policy library, plan document, or provider portal. Always use the source registry and payer website for the current authoritative policy.

## UnitedHealthcare commercial medical policy concept

UnitedHealthcare commercial medical and drug policies are used to assist in administering health benefits. Public policy pages describe that medical policies may be used to decide whether a health service is medically necessary or considered experimental, investigational, unproven, or not medically necessary. Coverage depends on the member-specific plan document and applicable laws, and prior authorization requirements may be listed separately from medical policies.

RAG mapping: medical necessity, unproven services, payer-specific plan document, prior authorization, documentation.

Suggested system behavior: retrieve this source when a claim reason involves high-cost service, prior authorization, medical necessity, or payer-policy review.

Source registry id: `uhc_commercial_medical_drug_policies`

## Aetna Clinical Policy Bulletin concept

Aetna Clinical Policy Bulletins describe services and procedures that Aetna considers medically necessary, cosmetic, experimental, investigational, or unproven. The public pages explain that CPBs are based on sources such as peer-reviewed literature, evidence-based consensus statements, expert opinions, and recognized healthcare-organization guidelines.

RAG mapping: medical necessity, experimental/unproven services, policy bulletin, supporting evidence.

Suggested system behavior: retrieve this source when a claim reason involves medical necessity, new technology, high-cost services, or documentation support.

Source registry id: `aetna_clinical_policy_bulletins`

## Cigna medical necessity concept

Cigna coverage policy pages describe coverage policies as tools for interpreting standard plan provisions. Public definitions describe medically necessary services as those that are clinically appropriate, consistent with generally accepted standards of medical practice, and not primarily for convenience or more costly than an equivalent alternative.

RAG mapping: medical necessity, clinical appropriateness, generally accepted standards, cost-effective alternative.

Suggested system behavior: retrieve this source when a claim reason involves high cost, unclear medical necessity, or procedure-diagnosis support.

Source registry id: `cigna_coverage_policies_medical_necessity`

## Anthem medical policy and clinical UM guideline concept

Anthem medical policies and Clinical UM guidelines are described as informational guidelines that support coverage decisions. Public pages note that medical policies address the medical need for services or procedures, and Clinical UM guidelines focus on selection criteria, length of stay, and location for generally accepted technologies or services. Coverage remains subject to applicable benefit plan terms and laws.

RAG mapping: utilization management, medical policy, medical necessity, documentation, site of service.

Suggested system behavior: retrieve this source when a claim reason involves UM review, prior authorization, site-of-service questions, or documentation requirements.

Source registry id: `anthem_medical_policies_clinical_um`
