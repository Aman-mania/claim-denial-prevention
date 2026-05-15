# Phase 7 UI Fix Notes

This patch fixes the role-aware Streamlit product UI after the first Phase 7 run.

## Fixes

- Analyst overview now includes a quick custom-claim workflow below the overview metrics.
- Developer overview now includes data/model readiness context and a small data preview.
- Product UI custom-claim form keys are unique and no longer collide with the embedded developer dashboard's Week 4 custom-claim builder.
- Policy evidence in the claim response is now compact: top policy sections are summarized first, while full retrieved evidence is placed in an expander.

## Why the duplicate-form error happened

Streamlit renders tab contents eagerly. The role-aware product UI rendered its own custom-claim form and also embedded the developer Risk Model tab, which has another custom-claim form. Both used the key `custom_claim_form`, causing a duplicate element-key failure. This patch changes the product UI forms to unique keys and parameterizes form instances.
