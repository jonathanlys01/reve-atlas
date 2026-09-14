<script lang="ts">
  import { Coordinator, wasmConnector } from "@uwdata/mosaic-core";
  import * as SQL from "@uwdata/mosaic-sql";
  import { EmbeddingAtlas } from "embedding-atlas/svelte";

  const DEFAULT_DATA_URL =
    "https://huggingface.co/datasets/jonathan-lys/reve-atlas/resolve/main/data/demo.parquet";
  const REQUIRED_COLUMNS = [
    "row_id",
    "window_id",
    "projection_x",
    "projection_y",
    "recon_loss",
    "big_recording_index",
    "session_index",
    "offset",
    "dataset",
    "modality",
    "sampling_rate",
    "n_channels",
  ]
    .map((column) => `"${column}"`)
    .join(", ");

  const datasetUrl = import.meta.env.VITE_ATLAS_DATA_URL ?? DEFAULT_DATA_URL;
  const coordinator = new Coordinator();
  const webGpuAvailable = "gpu" in navigator;

  async function initialize(): Promise<void> {
    const wasm = await wasmConnector();
    coordinator.databaseConnector(wasm);
    await coordinator.exec(`
      CREATE OR REPLACE TABLE dataset AS
      SELECT * FROM read_parquet(${SQL.literal(datasetUrl)})
    `);

    await coordinator.exec(`SELECT ${REQUIRED_COLUMNS} FROM dataset LIMIT 0`);
  }

  function describeError(error: unknown): string {
    const detail = error instanceof Error ? error.message : String(error);
    return [
      `Could not load the atlas dataset from ${datasetUrl}.`,
      "Check that the Hugging Face dataset is public and that the file URL supports CORS and HTTP range requests.",
      webGpuAvailable
        ? "WebGPU is available in this browser."
        : "WebGPU was not detected; use a current Chrome, Edge, or Safari release with WebGPU enabled.",
      `Details: ${detail}`,
    ].join(" ");
  }

  const initialized = initialize();
</script>

{#await initialized}
  <main class="status" aria-live="polite">
    <div class="status-card">
      <div class="spinner" aria-hidden="true"></div>
      <h1>REVE Embedding Atlas</h1>
      <p>Loading projected EEG and MEG windows from Hugging Face…</p>
    </div>
  </main>
{:then}
  <main class="atlas-shell">
    <EmbeddingAtlas
      {coordinator}
      data={{
        table: "dataset",
        id: "row_id",
        text: "window_id",
        projection: { x: "projection_x", y: "projection_y" },
      }}
      initialState={{
        layoutStates: {
          list: {
            showTable: true,
            showCharts: true,
            showEmbedding: true,
          },
        },
      }}
    />
  </main>
{:catch error}
  <main class="status error" role="alert">
    <div class="status-card">
      <p class="eyebrow">Dataset unavailable</p>
      <h1>REVE Embedding Atlas could not start</h1>
      <p>{describeError(error)}</p>
      <button type="button" onclick={() => window.location.reload()}>Retry</button>
    </div>
  </main>
{/await}

<style>
  .atlas-shell {
    width: 100%;
    height: 100%;
  }

  .status {
    box-sizing: border-box;
    display: grid;
    width: 100%;
    height: 100%;
    place-items: center;
    padding: 2rem;
    background:
      radial-gradient(circle at 20% 20%, rgb(86 101 255 / 14%), transparent 32rem),
      radial-gradient(circle at 80% 70%, rgb(34 197 175 / 12%), transparent 28rem), #f5f7fb;
  }

  .status-card {
    width: min(42rem, 100%);
    padding: 2.5rem;
    border: 1px solid #dfe5f0;
    border-radius: 1rem;
    background: rgb(255 255 255 / 92%);
    box-shadow: 0 1.25rem 4rem rgb(22 32 51 / 12%);
  }

  h1 {
    margin: 0 0 0.75rem;
    font-size: clamp(1.75rem, 4vw, 2.5rem);
    letter-spacing: -0.04em;
  }

  p {
    margin: 0;
    color: #526078;
    line-height: 1.6;
  }

  .eyebrow {
    margin-bottom: 0.5rem;
    color: #c2415a;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }

  .spinner {
    width: 1.75rem;
    height: 1.75rem;
    margin-bottom: 1.25rem;
    border: 3px solid #dce2ee;
    border-top-color: #5665ff;
    border-radius: 999px;
    animation: spin 0.8s linear infinite;
  }

  button {
    margin-top: 1.5rem;
    padding: 0.65rem 1rem;
    border: 0;
    border-radius: 0.5rem;
    color: white;
    background: #4c5cf4;
    cursor: pointer;
  }

  button:hover {
    background: #3949dc;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
