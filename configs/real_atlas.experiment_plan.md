# Real-data atlas experiment plan

## Projection

- Project all source windows to two dimensions with cosine UMAP.
- Fit UMAP on a deterministic seed-0 sample, then use deterministic cosine
  nearest-neighbor interpolation for the remaining windows so the full atlas
  remains reproducible and practical to build.
- Publish only `row_id`, projection coordinates, reconstruction loss, explicit
  identifiers, and the allowlisted recording metadata. Never publish embeddings.

## Selection experiments

- Use the deterministic `display.parquet` population as the full experiment
  population for the public demo.
- Run separate 1% and 10% retention experiments, independently for highest-loss
  (`top`) and lowest-loss (`bottom`) rankings within every
  `big_recording_index`; ties use `window_id` ascending.
- For each ranking configuration, run DPP selection inside each recording index
  using every display row in that index as the candidate pool. Derive the final
  cardinality as `max(1, ceil(selection_eta * candidate_pool_size))`.
- Sweep interaction weights for both multiplicative and additive quality-
  diversity kernels. Weight zero is the unweighted diversity baseline.
- Record selected memberships, marginal log-determinant gains, reconstruction
  loss summaries, Vendi scores, baseline overlap, adjacent-sweep overlap, and
  configuration/source fingerprints.

## Acceptance checks

Run `scripts.validate_real_artifacts` after generation. It checks the public
schema, absence of embeddings, finite and unique atlas rows, selection
membership, cardinalities, Vendi bounds, and manifest fingerprints.
