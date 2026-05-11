# Retrieval Analytics Visibility Fix

## Changes

- Replaced Plotly Express auto-colored vector scatter with explicit Plotly Graph Objects traces.
- Added a deterministic high-contrast color palette for policy sources.
- Added marker borders and a minimum marker size so policy chunks are visible on a white dashboard background.
- Added a dedicated selected-claim evidence overlay using larger diamond markers.
- Added a new **Source coverage** sub-tab that compares policy chunks present in the corpus against policy chunks actually retrieved as evidence.
- Added tests for source color mapping, source utilization coverage, and vector projection source preservation.

## Why

Some policy chunks were present and hoverable but almost invisible because auto-generated colors and marker sizing made them hard to see. Retrieval-only visuals also hid policy sources that existed in the corpus but were not selected as top evidence. This patch makes both situations explicit and demo-friendly.
