# REVE Embedding Atlas

Interactive visualization of projected REVE EEG and MEG windows using
[Apple Embedding Atlas](https://github.com/apple/embedding-atlas). The browser reads a public Parquet file directly
from Hugging Face through DuckDB-WASM; no application backend or client-side token is required.

## Local development

```bash
npm install
npm run dev
```

The default build uses the real atlas tables on Hugging Face. Override it for the synthetic demo or a local file:

```bash
VITE_ATLAS_DATA_URL=http://localhost:8000/demo.parquet npm run dev
```

The Parquet host must allow cross-origin `GET` requests and expose HTTP byte ranges.

## Regenerate and publish the demo data

```bash
uv sync --group dev
uv run python scripts/generate_dummy_data.py --output data/demo.parquet
uv run pytest
uv run python scripts/publish_hf.py
```

The publishing command uses the Hugging Face token configured on the machine, creates
`jonathan-lys/reve-atlas` as a public dataset if needed, and uploads the Parquet file and dataset card.



## Generate the real-data atlas

The real-data build reads the source embedding shards and an explicit recording-metadata
allowlist, then writes a public atlas table without embedding vectors plus ranking/DPP
selection tables. The checked-in example is a template; set the private input paths in a
copied config before running:

```bash
cp configs/real_atlas.example.yaml /tmp/real_atlas.yaml
# edit input.embeddings, input.recording_metadata, and output.directory
PYTHONPATH=. uv run python -m scripts.run_real_atlas --config /tmp/real_atlas.yaml
```

The output directory contains `atlas.parquet`, `selections.parquet`,
`selection_runs.parquet`, `manifest.json`, and validation logs. The browser uses `display.parquet`, a deterministic sub-2M display table that retains every row referenced by the selection tables; the full `atlas.parquet` remains available for offline use. The manifest records
the source/configuration fingerprints, projection provenance, and selection sweep counts.
The validator can be rerun independently:

```bash
PYTHONPATH=. uv run python -m scripts.validate_real_artifacts --config /tmp/real_atlas.yaml
```

For the demo selection sweep, treat the deterministic `display.parquet` population as the full candidate set. Extract its private embeddings, then run the display config separately (the private cache is never uploaded):

```bash
uv run python scripts/build_display_selection_source.py --source /path/to/embedding_shards --display data/real/display.parquet --output data/real/display_selection_source.parquet
PYTHONPATH=. uv run python -m scripts.build_selections --config /tmp/real_atlas_display.yaml
PYTHONPATH=. uv run python -m scripts.validate_real_artifacts --config /tmp/real_atlas_display.yaml
```

This produces separate 1% and 10% experiments, stratified within all recording indices. Each DPP uses every display row in its recording index as its candidate pool and sweeps both kernel interaction parameters.

For the real-data frontend, configure all three public table URLs at build time:

```bash
VITE_ATLAS_ATLAS_URL=https://huggingface.co/datasets/ORG/REPO/resolve/main/data/atlas.parquet \
VITE_ATLAS_SELECTIONS_URL=https://huggingface.co/datasets/ORG/REPO/resolve/main/data/selections.parquet \
VITE_ATLAS_RUNS_URL=https://huggingface.co/datasets/ORG/REPO/resolve/main/data/selection_runs.parquet \
npm run build
```

Publish the validated real-data tables with:

```bash
uv run python scripts/publish_real_hf.py --repo-id jonathan-lys/reve-atlas --data-dir data/real --card hf/README.real.md
```

## Checks

```bash
npm run check
npm run build
uv run pytest
uv run ruff check scripts tests
```

## Publish the standalone repository

The source is maintained at `viz/atlas/` in `reve-2` and mirrored to the standalone repository with a subtree split:

```bash
git subtree split --prefix viz/atlas -b codex/reve-atlas-publish
git push git@github.com:jonathanlys01/reve-atlas.git codex/reve-atlas-publish:main
git branch -D codex/reve-atlas-publish
```

GitHub Pages deploys automatically from `main`. The repository's Pages source must be set to GitHub Actions once:

```bash
gh api --method POST repos/jonathanlys01/reve-atlas/pages -f build_type=workflow
```

The deployed application is available at <https://jonathanlys01.github.io/reve-atlas/>.

## Import a full-corpus clustered DPP selection

Clustered DPP is a separate strategy from the per-recording sweeps. Its manifest
covers the full embedding corpus; the browser overlays only its intersection with
the existing display population. Full and displayed counts are shown separately,
and the complete sorted window manifest is downloadable.

Use `atlas.parquet` and `display.parquet` from the same pinned Hugging Face revision.
The selector must have finished successfully, with a `.txt` manifest and its
sibling `.json` report. Run the bounded-memory importer on scratch storage:

```bash
uv run python scripts/import_clustered_selection.py \
  --manifest /path/to/clustered_dpp_n1000_k100_seed29.txt \
  --atlas /path/to/atlas.parquet --display /path/to/display.parquet \
  --output-dir /path/to/new-clustered-overlay --work-dir /path/to/scratch \
  --atlas-revision SOURCE_HF_COMMIT
```

The importer checks the checksum, exact retention quota, sorted unique IDs, full
atlas population, and membership of every selected/display ID in the full atlas.
It refuses to overwrite an existing output directory. Its three outputs contain
only IDs and allowlisted provenance, never private embedding vectors or paths:

- `clustered_manifest.txt`: all selected IDs.
- `clustered_selections.parquet`: selected IDs present in the display file.
- `clustered_run.json`: full/display counts, configuration and hashes.

With write access to the dataset, publish those three files together in one Hugging
Face commit under `data/`. Preserve the existing atlas, display and sweep files.
The default frontend automatically loads `data/clustered_run.json`. For another
host, set `VITE_ATLAS_CLUSTERED_RUN_URL` to that metadata URL; the two companion
files must be in the same directory. Use an absolute URL. A missing or invalid
overlay leaves the existing atlas usable; invalid metadata or memberships are
reported beside the selection controls. Custom atlas URLs disable the default
overlay unless explicitly configured.

Additional checks:

```bash
node --test tests/clustered_run.test.mjs
uv run pytest tests/test_import_clustered_selection.py
```
