---
pretty_name: REVE Embedding Atlas
license: mit
task_categories:
  - feature-extraction
tags:
  - eeg
  - embeddings
  - visualization
  - parquet
---

# REVE Embedding Atlas

This dataset contains precomputed two-dimensional coordinates and approved metadata
for REVE EEG windows, together with reconstruction-loss ranking and diversity-selection
tables used by the [REVE Embedding Atlas](https://jonathanlys01.github.io/reve-atlas/).

The public tables contain no original 512-dimensional embedding vectors. Projection and
selection provenance, source fingerprints, and schemas are recorded in `data/manifest.json`.

The demo display table is the full population used for selection. The experiments retain 1% and 10% of each `big_recording_index`, separately for high- and low-reconstruction-loss rankings. DPP runs use every display row in the relevant recording index as their candidate pool and sweep multiplicative and additive interaction parameters.
