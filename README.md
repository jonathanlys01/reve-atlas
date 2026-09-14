# REVE Embedding Atlas

Interactive visualization of projected REVE EEG and MEG windows using
[Apple Embedding Atlas](https://github.com/apple/embedding-atlas). The browser reads a public Parquet file directly
from Hugging Face through DuckDB-WASM; no application backend or client-side token is required.

## Local development

```bash
npm install
npm run dev
```

The default dataset is
`https://huggingface.co/datasets/jonathan-lys/reve-atlas/resolve/main/data/demo.parquet`. Override it for local testing:

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
