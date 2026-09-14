# Real-data atlas experiment plan

## Projection

- Project all source windows to two dimensions with cosine UMAP.
- Fit UMAP on a deterministic seed-0 sample, then use deterministic cosine
  nearest-neighbor interpolation for the remaining windows so the full atlas
  remains reproducible and practical to build.
- Publish only `row_id`, projection coordinates, reconstruction loss, explicit
  identifiers, and the allowlisted recording metadata. Never publish embeddings.

## Selection experiments

- Rank globally and within each recording in both directions: lowest-loss
  (“top”) and highest-loss (“bottom”). Use candidate pools of 5,000 globally
  and 256 per recording, with `window_id` ascending as the exact tie-break.
- For recordings 42 and 442, select 32 windows from each 256-window candidate
  pool with all-start greedy MAP DPP.
- Sweep interaction weights for both multiplicative and additive quality-
  diversity kernels. Weight zero is the unweighted diversity baseline.
- Record selected memberships, marginal log-determinant gains, reconstruction
  loss summaries, Vendi scores, baseline overlap, adjacent-sweep overlap, and
  configuration/source fingerprints.

## Acceptance checks

Run `scripts.validate_real_artifacts` after generation. It checks the public
schema, absence of embeddings, finite and unique atlas rows, selection
membership, cardinalities, Vendi bounds, and manifest fingerprints.
