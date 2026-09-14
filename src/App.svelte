<script lang="ts">
  import { Coordinator, wasmConnector } from "@uwdata/mosaic-core";
  import * as SQL from "@uwdata/mosaic-sql";
  import { tick } from "svelte";
  import { EmbeddingAtlas } from "embedding-atlas/svelte";

  type Method = "ranking" | "dpp";
  type Scope = "global" | "per_recording";
  type ViewMode = "all" | "candidate" | "selected";
  type Run = {
    run_id: string;
    config_id: string;
    method: Method;
    direction: "top" | "bottom";
    scope: Scope;
    candidate_run_id: string | null;
    big_recording_index: number | null;
    candidate_k: number;
    selection_eta: number;
    actual_size: number;
    kernel_method: "multiplicative" | "additive" | null;
    w_interaction: number | null;
    vendi_score: number;
    log_det: number | null;
    mean_recon_loss: number;
    median_recon_loss: number;
    min_recon_loss: number;
    max_recon_loss: number;
    mean_utility: number;
    baseline_jaccard: number | null;
    adjacent_jaccard: number | null;
  };

  const DEFAULT_ATLAS_URL =
    "https://huggingface.co/datasets/jonathan-lys/reve-atlas/resolve/main/data/display.parquet";
  const DEFAULT_SELECTIONS_URL =
    "https://huggingface.co/datasets/jonathan-lys/reve-atlas/resolve/main/data/selections.parquet";
  const DEFAULT_RUNS_URL =
    "https://huggingface.co/datasets/jonathan-lys/reve-atlas/resolve/main/data/selection_runs.parquet";
  const atlasUrl =
    import.meta.env.VITE_ATLAS_ATLAS_URL ?? import.meta.env.VITE_ATLAS_DATA_URL ?? DEFAULT_ATLAS_URL;
  const selectionsUrl = import.meta.env.VITE_ATLAS_SELECTIONS_URL ?? DEFAULT_SELECTIONS_URL;
  const runsUrl = import.meta.env.VITE_ATLAS_RUNS_URL ?? DEFAULT_RUNS_URL;
  const REQUIRED_COLUMNS = [
    "row_id", "window_id", "projection_x", "projection_y", "recon_loss",
    "big_recording_index", "session_index", "offset", "dataset", "modality",
    "sampling_rate", "n_channels",
  ].map((column) => `"${column}"`).join(", ");
  const DISPLAY_POINT_CAP = 2_000_000;
  const DISPLAY_SAMPLE_CAP = 1_760_000;
  const coordinator = new Coordinator();
  const webGpuAvailable = "gpu" in navigator;

  let realMode = false;
  let runs: Run[] = [];
  let activeRunId = "";
  let activeRun: Run | null = null;
  let validRuns: Run[] = [];
  let availableRecordings: string[] = [];
  let tradeoffRuns: Run[] = [];
  let selectedMethod: Method = "ranking";
  let selectedDirection: "top" | "bottom" = "top";
  let selectedScope: Scope = "global";
  let selectedRecording = "all";
  let selectedKernel: "multiplicative" | "additive" = "multiplicative";
  let viewMode: ViewMode = "all";
  let activeTableVersion = 0;
  let refreshSerial = 0;
  let refreshChain: Promise<void> = Promise.resolve();

  $: availableRecordings = [
    "all",
    ...new Set(
      runs.filter((run) => run.method === selectedMethod &&
        run.direction === selectedDirection &&
        run.scope === (selectedMethod === "dpp" ? "per_recording" : selectedScope) &&
        (selectedMethod === "ranking" || run.kernel_method === selectedKernel) &&
        run.big_recording_index !== null)
        .map((run) => run.big_recording_index as number)
        .sort((a, b) => a - b).map(String),
    ),
  ];
  $: validRuns = runs.filter((run) => {
    const recordingMatches = selectedScope === "global"
      ? run.big_recording_index === null
      : selectedRecording === "all" || String(run.big_recording_index) === selectedRecording;
    return run.method === selectedMethod && run.direction === selectedDirection &&
      run.scope === selectedScope && recordingMatches &&
      (selectedMethod === "ranking" || run.kernel_method === selectedKernel);
  });
  $: activeRun = validRuns.find((run) => run.run_id === activeRunId) ?? validRuns[0] ?? null;
  function sameSelectionSweep(left: Run, right: Run): boolean {
    return left.method === right.method && left.config_id === right.config_id &&
      left.direction === right.direction && left.scope === right.scope &&
      left.kernel_method === right.kernel_method && left.w_interaction === right.w_interaction;
  }
  $: interactionRuns = selectedRecording === "all" && selectedScope === "per_recording"
    ? validRuns.filter((run, index, all) => all.findIndex((candidate) => sameSelectionSweep(candidate, run)) === index)
    : validRuns;
  function runsForActiveView(run: Run): Run[] {
    if (run.scope !== "per_recording" || selectedRecording !== "all") return [run];
    const matchingRuns = validRuns.filter((candidate) => sameSelectionSweep(candidate, run));
    return matchingRuns.length ? matchingRuns : [run];
  }
  $: activeViewRuns = activeRun ? runsForActiveView(activeRun) : [];
  $: activeViewCandidateK = activeViewRuns.reduce((total, run) => total + run.candidate_k, 0);
  $: activeViewSize = activeViewRuns.reduce((total, run) => total + run.actual_size, 0);
  $: tradeoffRuns = activeRun?.method === "dpp"
    ? runs.filter((run) => run.method === "dpp" && run.direction === activeRun?.direction &&
        run.scope === activeRun?.scope && run.big_recording_index === activeRun?.big_recording_index &&
        run.config_id === activeRun?.config_id)
        .sort((a, b) => (a.w_interaction ?? 0) - (b.w_interaction ?? 0))
    : [];
  $: vendiMin = tradeoffRuns.length ? Math.min(...tradeoffRuns.map((run) => run.vendi_score)) : 0;
  $: vendiMax = tradeoffRuns.length ? Math.max(...tradeoffRuns.map((run) => run.vendi_score)) : 1;
  $: lossMin = tradeoffRuns.length ? Math.min(...tradeoffRuns.map((run) => run.mean_recon_loss)) : 0;
  $: lossMax = tradeoffRuns.length ? Math.max(...tradeoffRuns.map((run) => run.mean_recon_loss)) : 1;

  async function initialize(): Promise<void> {
    const wasm = await wasmConnector();
    coordinator.databaseConnector(wasm);
    await coordinator.exec(`CREATE OR REPLACE VIEW atlas_source AS SELECT * FROM read_parquet(${SQL.literal(atlasUrl)})`);
    realMode = Boolean(selectionsUrl && runsUrl);
    if (!realMode) {
      await coordinator.exec(`
        CREATE OR REPLACE TABLE display_points AS
        SELECT * FROM atlas_source
        WHERE hash(row_id) % 100 < 8
        LIMIT ${DISPLAY_POINT_CAP}
      `);
      await coordinator.exec(`SELECT ${REQUIRED_COLUMNS} FROM display_points LIMIT 0`);
      await coordinator.exec("CREATE OR REPLACE VIEW dataset AS SELECT * FROM display_points");
      return;
    }
    await coordinator.exec(`CREATE OR REPLACE TABLE selection_memberships AS SELECT * FROM read_parquet(${SQL.literal(selectionsUrl)})`);
    await coordinator.exec(`CREATE OR REPLACE TABLE selection_runs AS SELECT * FROM read_parquet(${SQL.literal(runsUrl)})`);
    await coordinator.exec(`
      CREATE OR REPLACE TABLE display_points AS
      WITH required_points AS (
        SELECT DISTINCT atlas_source.*
        FROM atlas_source
        INNER JOIN (SELECT DISTINCT row_id FROM selection_memberships) AS members USING (row_id)
      ), sampled_points AS (
        SELECT atlas_source.*
        FROM atlas_source
        WHERE hash(atlas_source.row_id) % 100 < 8
          AND NOT EXISTS (SELECT 1 FROM required_points WHERE required_points.row_id = atlas_source.row_id)
        LIMIT ${DISPLAY_SAMPLE_CAP}
      )
      SELECT * FROM required_points
      UNION ALL
      SELECT * FROM sampled_points
    `);
    await coordinator.exec(`SELECT ${REQUIRED_COLUMNS} FROM display_points LIMIT 0`);
    runs = (await coordinator.query(
      "SELECT * FROM selection_runs ORDER BY method, direction, scope, big_recording_index NULLS FIRST, run_id",
      { type: "json" },
    )) as Run[];
    if (!runs.length) throw new Error("The selection-runs table is empty.");
    const first = runs.find((run) => run.method === "ranking" && run.direction === "top" && run.scope === "global") ?? runs[0];
    activeRunId = first.run_id;
    await refreshActiveView(first);
  }

  async function refreshActiveView(run: Run | null = activeRun): Promise<void> {
    const requestId = ++refreshSerial;
    refreshChain = refreshChain.catch(() => undefined).then(async () => {
      if (!realMode || !run || requestId !== refreshSerial) return;
      const viewRuns = runsForActiveView(run);
      const selectedRuns = viewRuns.map((item) => SQL.literal(item.run_id)).join(", ");
      const candidateRuns = viewRuns
        .map((item) => SQL.literal(item.method === "dpp" ? item.candidate_run_id ?? item.run_id : item.run_id))
        .join(", ");
      const filter = viewMode === "selected" ? "WHERE selected_members.row_id IS NOT NULL" :
        viewMode === "candidate" ? "WHERE candidate_members.row_id IS NOT NULL" : "";
      await coordinator.exec(`
      CREATE OR REPLACE VIEW active_points AS
      WITH selected_members AS (
        SELECT row_id, rank AS selected_rank FROM selection_memberships WHERE run_id IN (${selectedRuns})
      ), candidate_members AS (
        SELECT row_id, rank AS candidate_rank FROM selection_memberships WHERE run_id IN (${candidateRuns})
      )
      SELECT display_points.*, selected_members.selected_rank, candidate_members.candidate_rank,
        selected_members.row_id IS NOT NULL AS selected,
        candidate_members.row_id IS NOT NULL AS candidate,
        CASE
          WHEN selected_members.row_id IS NOT NULL THEN 2
          WHEN candidate_members.row_id IS NOT NULL THEN 1
          ELSE 0
        END::INTEGER AS display_category
      FROM display_points
      LEFT JOIN selected_members USING (row_id)
      LEFT JOIN candidate_members USING (row_id)
      ${filter}
    `);
      if (requestId === refreshSerial) activeTableVersion += 1;
    });
    await refreshChain;
  }

  async function onFilterChange(): Promise<void> {
    if (selectedMethod === "dpp") selectedScope = "per_recording";
    if (selectedScope === "global") selectedRecording = "all";
    await tick();
    if (selectedScope === "per_recording" && selectedRecording !== "all" && !availableRecordings.includes(selectedRecording)) {
      selectedRecording = "all";
      await tick();
    }
    const next = validRuns[0] ?? null;
    activeRunId = next?.run_id ?? "";
    await refreshActiveView(next);
  }

  function formatNumber(value: number | null): string {
    return value === null || !Number.isFinite(value) ? "—" : value.toPrecision(5);
  }

  function describeError(error: unknown): string {
    const detail = error instanceof Error ? error.message : String(error);
    return [`Could not load the atlas dataset from ${atlasUrl}.`,
      "Check that the public Parquet files support CORS and HTTP range requests.",
      webGpuAvailable ? "WebGPU is available." : "WebGPU was not detected; precomputed coordinates remain supported.",
      `Details: ${detail}`].join(" ");
  }

  const initialized = initialize();
</script>

{#await initialized}
  <main class="status" aria-live="polite"><div class="status-card"><div class="spinner" aria-hidden="true"></div><h1>REVE Embedding Atlas</h1><p>Loading projected EEG and MEG windows…</p></div></main>
{:then}
  <main class:real-mode={realMode} class="atlas-app">
    {#if realMode}
      <aside class="control-rail" aria-label="Atlas controls">
        <p class="eyebrow">REVE explorer</p>
        <h1>Embedding Atlas</h1>
        <p class="rail-copy">Switch among precomputed rankings and diversity selections.</p>
        <p class="display-note">Display cap: about 2M points; all selection members are retained. Per-recording views union matching intra-index runs.</p>
        <div class="control-stack">
          <label>Method<select bind:value={selectedMethod} onchange={onFilterChange}><option value="ranking">Ranking</option><option value="dpp">DPP selection</option></select></label>
          <label>Direction<select bind:value={selectedDirection} onchange={onFilterChange}><option value="top">Top loss</option><option value="bottom">Bottom loss</option></select></label>
          <label>Scope<select bind:value={selectedScope} onchange={onFilterChange}><option value="global">Global</option><option value="per_recording">Per recording</option></select></label>
          <label>Recording<select bind:value={selectedRecording} onchange={onFilterChange} disabled={selectedScope === "global"}>{#each availableRecordings as recording}<option value={recording}>{recording === "all" ? "All indexed recordings" : recording}</option>{/each}</select></label>
          <label>Kernel<select bind:value={selectedKernel} onchange={onFilterChange} disabled={selectedMethod === "ranking"}><option value="multiplicative">Multiplicative · utility ↑</option><option value="additive">Additive · diversity ↑</option></select></label>
          <label>Selection sweep<select bind:value={activeRunId} onchange={() => refreshActiveView(activeRun)}>{#each interactionRuns as run}<option value={run.run_id}>{run.method === "ranking" ? `${run.config_id} · ${run.actual_size} rows` : `w=${run.w_interaction} · ${run.actual_size} rows`}</option>{/each}</select></label>
          <label>View<select bind:value={viewMode} onchange={() => refreshActiveView(activeRun)}><option value="all">All atlas points</option><option value="candidate">Candidate pool</option><option value="selected">Selected rows only</option></select></label>
        </div>
      </aside>
    {/if}
    <section class="atlas-shell">
      {#key realMode ? activeTableVersion : "demo"}
        <EmbeddingAtlas
          {coordinator}
          data={{ table: realMode ? "active_points" : "dataset", id: "row_id", text: "window_id", projection: { x: "projection_x", y: "projection_y" } }}
          defaultChartsConfig={{
            include: realMode ? ["recon_loss", "dataset", "modality", "big_recording_index", "selected", "candidate"] : ["recon_loss"],
            embedding: { data: { x: "projection_x", y: "projection_y", text: "window_id", category: realMode ? "display_category" : null } },
            table: true,
          }}
          chartTheme={{ categoryColors: ["#94a3b8", "#f59e0b", "#e11d48"] }}
          embeddingViewConfig={{ downsampleMaxPoints: DISPLAY_POINT_CAP, pointSize: 1.5 }}
        />
      {/key}
    </section>
    {#if realMode && activeRun}
      <aside class="metrics-rail" aria-label="Active run summary">
        <p class="eyebrow">Active run</p>
        <h2>{activeRun.method === "dpp" ? "DPP selection" : "Loss ranking"}</h2>
        <p class="run-id">{activeRun.run_id}</p>
        {#if activeViewRuns.length > 1}<p class="view-note">Showing {activeViewRuns.length} intra-index runs in this view.</p>{/if}
        <div class="metric-grid">
          <div class="metric hero"><strong>{formatNumber(activeRun.vendi_score)}</strong><span>Vendi diversity</span></div>
          <div class="metric"><strong>{formatNumber(activeRun.log_det)}</strong><span>log determinant</span></div>
          <div class="metric"><strong>{activeViewCandidateK}</strong><span>candidate rows in view</span></div>
          <div class="metric"><strong>{activeViewSize}</strong><span>selected rows in view</span></div>
          <div class="metric"><strong>{formatNumber(activeRun.mean_recon_loss)}</strong><span>mean loss</span></div>
          <div class="metric"><strong>{formatNumber(activeRun.median_recon_loss)}</strong><span>median loss</span></div>
          {#if activeRun.method === "dpp"}<div class="metric"><strong>{formatNumber(activeRun.baseline_jaccard)}</strong><span>baseline Jaccard</span></div><div class="metric"><strong>{formatNumber(activeRun.adjacent_jaccard)}</strong><span>adjacent Jaccard</span></div>{/if}
        </div>
        {#if tradeoffRuns.length > 1}
          <div class="tradeoff"><h3>Vendi / loss sweep</h3><svg viewBox="0 0 220 130" role="img" aria-label="Vendi versus reconstruction loss trade-off"><line x1="22" y1="8" x2="22" y2="108" /><line x1="22" y1="108" x2="214" y2="108" /><polyline points={tradeoffRuns.map((run) => `${22 + ((run.vendi_score - vendiMin) / Math.max(vendiMax - vendiMin, 1e-9)) * 192},${108 - ((run.mean_recon_loss - lossMin) / Math.max(lossMax - lossMin, 1e-9)) * 96}`).join(" ")} />{#each tradeoffRuns as run}<circle cx={22 + ((run.vendi_score - vendiMin) / Math.max(vendiMax - vendiMin, 1e-9)) * 192} cy={108 - ((run.mean_recon_loss - lossMin) / Math.max(lossMax - lossMin, 1e-9)) * 96} r={run.run_id === activeRun.run_id ? 5 : 3} class:current={run.run_id === activeRun.run_id} />{/each}</svg><div class="axis-labels"><span>Vendi →</span><span>loss ↑</span></div><small>{activeRun.kernel_method === "multiplicative" ? "Higher w emphasizes ranking utility." : "Higher w emphasizes interaction/diversity."}</small></div>
        {/if}
      </aside>
    {/if}
  </main>
{:catch error}
  <main class="status error" role="alert"><div class="status-card"><p class="eyebrow">Dataset unavailable</p><h1>REVE Embedding Atlas could not start</h1><p>{describeError(error)}</p><button type="button" onclick={() => window.location.reload()}>Retry</button></div></main>
{/await}

<style>
  .atlas-app { display: grid; grid-template-columns: 17rem minmax(0, 1fr) 17rem; min-width: 0; min-height: 0; width: 100%; height: 100%; overflow: hidden; background: #f5f7fb; }
  .atlas-app:not(.real-mode) { display: block; }
  .atlas-shell { min-width: 0; min-height: 0; width: 100%; height: 100%; overflow: hidden; position: relative; }
  .control-rail, .metrics-rail { z-index: 2; box-sizing: border-box; min-width: 0; min-height: 0; overflow: auto; padding: 1.25rem 1rem; border-color: #dfe5f0; background: rgb(255 255 255 / 96%); }
  .control-rail { border-right: 1px solid #dfe5f0; }
  .metrics-rail { border-left: 1px solid #dfe5f0; }
  h1 { margin: 0 0 0.55rem; font-size: 1.45rem; letter-spacing: -0.04em; }
  h2 { margin: 0.1rem 0 0.3rem; font-size: 1.15rem; }
  h3 { margin: 1.25rem 0 0.25rem; color: #526078; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }
  p { margin: 0; color: #526078; line-height: 1.45; }
  .rail-copy { margin-bottom: 0.35rem; font-size: 0.78rem; }
  .display-note { margin-bottom: 1.25rem; color: #7c879b; font-size: 0.68rem; }
  .eyebrow { margin-bottom: 0.45rem; color: #c2415a; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; }
  .control-stack { display: grid; gap: 0.75rem; }
  label { display: grid; gap: 0.25rem; color: #526078; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }
  select { width: 100%; min-width: 0; box-sizing: border-box; padding: 0.45rem 0.4rem; border: 1px solid #d4dbea; border-radius: 0.35rem; color: #172033; background: #fff; font-size: 0.75rem; text-transform: none; }
  select:disabled { color: #8b95a8; background: #f3f5f9; }
  .run-id { overflow: hidden; margin-bottom: 0.45rem; color: #8792a7; font: 0.6rem ui-monospace, monospace; text-overflow: ellipsis; white-space: nowrap; }
  .view-note { margin-bottom: 1rem; color: #69758b; font-size: 0.68rem; }
  .metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem 0.55rem; }
  .metric { display: grid; gap: 0.1rem; }
  .metric strong { color: #202d91; font-size: 0.92rem; }
  .metric span, small, .axis-labels { color: #69758b; font-size: 0.65rem; }
  .metric.hero { grid-column: 1 / -1; padding: 0.7rem; border-radius: 0.45rem; background: #eef0ff; }
  .metric.hero strong { font-size: 1.5rem; }
  .tradeoff svg { display: block; width: 100%; height: 8rem; overflow: visible; }
  .tradeoff line { stroke: #cbd3e2; stroke-width: 1; }
  .tradeoff polyline { fill: none; stroke: #5665ff; stroke-width: 1.5; }
  .tradeoff circle { fill: #5665ff; }
  .tradeoff circle.current { fill: #c2415a; stroke: white; stroke-width: 1.5; }
  .axis-labels { display: flex; justify-content: space-between; }
  .status { box-sizing: border-box; display: grid; width: 100%; height: 100%; place-items: center; padding: 2rem; background: radial-gradient(circle at 20% 20%, rgb(86 101 255 / 14%), transparent 32rem), radial-gradient(circle at 80% 70%, rgb(34 197 175 / 12%), transparent 28rem), #f5f7fb; }
  .status-card { width: min(42rem, 100%); padding: 2.5rem; border: 1px solid #dfe5f0; border-radius: 1rem; background: rgb(255 255 255 / 92%); box-shadow: 0 1.25rem 4rem rgb(22 32 51 / 12%); }
  .spinner { width: 1.75rem; height: 1.75rem; margin-bottom: 1.25rem; border: 3px solid #dce2ee; border-top-color: #5665ff; border-radius: 999px; animation: spin 0.8s linear infinite; }
  button { margin-top: 1.5rem; padding: 0.65rem 1rem; border: 0; border-radius: 0.5rem; color: white; background: #4c5cf4; cursor: pointer; }
  @media (max-width: 1050px) { .atlas-app.real-mode { grid-template-columns: 14rem minmax(0, 1fr); grid-template-rows: minmax(0, 1fr) auto; } .metrics-rail { grid-column: 1 / -1; display: flex; gap: 1rem; align-items: start; border-top: 1px solid #dfe5f0; border-left: 0; } .metric-grid { display: flex; flex-wrap: wrap; } .metric.hero { min-width: 8rem; } .tradeoff { width: 14rem; } }
  @media (max-width: 700px) { .atlas-app.real-mode { display: block; overflow: auto; } .control-rail, .metrics-rail { border: 0; border-bottom: 1px solid #dfe5f0; } .atlas-shell { height: 70vh; min-height: 32rem; } .metrics-rail { display: block; } }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
