# Retrieval Analytics contract fix

## Fixed

- Restored backward-compatible return behavior for `_prepare_reason_source_flow(matches)`.
- The function now returns only a DataFrame by default, matching the original test/helper contract.
- The renderer opts into the extended `(flow, all_sources)` return using `include_all_sources=True`.
- Added an idempotent dashboard hardening script to replace deprecated `use_container_width` usage across `dev_dashboard/`.

## Why

The previous source-visibility update changed `_prepare_reason_source_flow()` to return a tuple unconditionally. Existing tests and callers expected a DataFrame, which caused `AttributeError: 'tuple' object has no attribute 'columns'`.
