---
document_id: official_policy_source_registry_summary
source_type: source_registry_summary
tags: [policy_sources, cms, hhs, payer, hipaa, ncci, medical_necessity]
---

# Official Policy Source Registry Summary

This document lists authoritative policy sources used by the Week 6 RAG corpus. It helps retrieval return source-level guidance even when full PDFs/pages are not downloaded locally.

## CMS National Correct Coding Initiative (NCCI) Overview

Source ID: `cms_ncci_overview`
Publisher: Centers for Medicare & Medicaid Services
Source type: official_government
URL: https://www.cms.gov/medicare-medicaid-coordination/national-correct-coding-initiative-ncci
Policy tags: cms, ncci, coding, improper_payment, claim_edit

## Medicare NCCI Policy Manual

Source ID: `cms_ncci_policy_manual`
Publisher: Centers for Medicare & Medicaid Services
Source type: official_government
URL: https://www.cms.gov/medicare/coding-billing/national-correct-coding-initiative-ncci-edits/medicare-ncci-policy-manual
Policy tags: cms, ncci, coding, procedure, cpt, hcpcs, policy_manual

## Medicare NCCI Edits

Source ID: `cms_ncci_medicare_edits`
Publisher: Centers for Medicare & Medicaid Services
Source type: official_government
URL: https://www.cms.gov/medicare-medicaid-coordination/national-correct-coding-initiative-ncci/ncci-medicare
Policy tags: cms, ncci, ptp, mue, coding_edit, procedure_combination

## Prior Authorization and Pre-Claim Review Initiatives

Source ID: `cms_prior_auth_preclaim`
Publisher: Centers for Medicare & Medicaid Services
Source type: official_government
URL: https://www.cms.gov/Research-Statistics-Data-and-Systems/Monitoring-Programs/Medicare-FFS-Compliance-Programs/Prior-Authorization-Initiatives/Prior-Authorization-of-Non-emergent-Hyperbaric-Oxygen.html
Policy tags: cms, prior_authorization, pre_claim_review, documentation, medical_necessity

## Medicare Coverage Center

Source ID: `cms_coverage_center`
Publisher: Centers for Medicare & Medicaid Services
Source type: official_government
URL: https://www.cms.gov/medicare/coverage/center
Policy tags: cms, coverage, ncd, lcd, medical_necessity

## Local Coverage Determinations (LCDs)

Source ID: `cms_local_coverage_determinations`
Publisher: Centers for Medicare & Medicaid Services
Source type: official_government
URL: https://www.cms.gov/Medicare/Coverage/DeterminationProcess/LCDs.html
Policy tags: cms, lcd, local_coverage, medical_necessity, mac

## Medicare Coverage Database Search

Source ID: `cms_medicare_coverage_database`
Publisher: Centers for Medicare & Medicaid Services
Source type: official_government
URL: https://www.cms.gov/medicare-coverage-database/search.aspx
Policy tags: cms, mcd, ncd, lcd, article, coding_article

## HIPAA Privacy Rule

Source ID: `hhs_hipaa_privacy_rule`
Publisher: U.S. Department of Health and Human Services
Source type: official_government
URL: https://www.hhs.gov/hipaa/for-professionals/privacy/index.html
Policy tags: hhs, hipaa, privacy, phi, minimum_necessary, claim_processing

## HIPAA Security Rule

Source ID: `hhs_hipaa_security_rule`
Publisher: U.S. Department of Health and Human Services
Source type: official_government
URL: https://www.hhs.gov/hipaa/for-professionals/security/index.html
Policy tags: hhs, hipaa, security, ephi, safeguards, audit_logs

## HIPAA Breach Notification Rule

Source ID: `hhs_hipaa_breach_notification_rule`
Publisher: U.S. Department of Health and Human Services
Source type: official_government
URL: https://www.hhs.gov/hipaa/for-professionals/breach-notification/index.html
Policy tags: hhs, hipaa, breach, notification, phi, security_incident

## HIPAA Enforcement Rule

Source ID: `hhs_hipaa_enforcement_rule`
Publisher: U.S. Department of Health and Human Services
Source type: official_government
URL: https://www.hhs.gov/hipaa/for-professionals/special-topics/enforcement-rule/index.html
Policy tags: hhs, hipaa, enforcement, compliance, penalty

## UnitedHealthcare Commercial Medical & Drug Policies

Source ID: `uhc_commercial_medical_drug_policies`
Publisher: UnitedHealthcare
Source type: public_payer_reference
URL: https://www.uhcprovider.com/en/policies-protocols/commercial-policies/commercial-medical-drug-policies.html
Policy tags: payer, unitedhealthcare, medical_policy, medical_necessity, prior_authorization

## Aetna Clinical Policy Bulletins

Source ID: `aetna_clinical_policy_bulletins`
Publisher: Aetna
Source type: public_payer_reference
URL: https://www.aetna.com/health-care-professionals/clinical-policy-bulletins.html
Policy tags: payer, aetna, clinical_policy_bulletin, medical_necessity, experimental_unproven

## Cigna Coverage Policies and Medical Necessity Definitions

Source ID: `cigna_coverage_policies_medical_necessity`
Publisher: Cigna Healthcare
Source type: public_payer_reference
URL: https://www.cigna.com/health-care-providers/coverage-and-claims/policies/medical-necessity-definitions
Policy tags: payer, cigna, coverage_policy, medical_necessity, plan_document

## Anthem Medical Policies and Clinical UM Guidelines

Source ID: `anthem_medical_policies_clinical_um`
Publisher: Anthem
Source type: public_payer_reference
URL: https://www.anthem.com/provider/policies/clinical-guidelines/
Policy tags: payer, anthem, medical_policy, clinical_um, medical_necessity, documentation

## Internal Claim Validation Rules for Demo Dataset

Source ID: `internal_claim_validation_rules`
Publisher: Claim Denial Prevention Project
Source type: internal_project_policy
URL: policy_docs/rag_seed/us_healthcare_claim_policy_seed_pack.md
Policy tags: internal, claim_completeness, demo_dataset, diagnosis, procedure, billed_amount, documentation
