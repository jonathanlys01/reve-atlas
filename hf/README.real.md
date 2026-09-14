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
