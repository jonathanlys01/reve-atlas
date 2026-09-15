<script lang="ts">
  import { Coordinator, wasmConnector } from "@uwdata/mosaic-core";
  import * as SQL from "@uwdata/mosaic-sql";
  import { tick } from "svelte";
  import { EmbeddingAtlas } from "embedding-atlas/svelte";

  type Method = "ranking" | "dpp";
  type Scope = "global" | "per_recording";
  type ViewMode = "all" | "candidate" | "selected";
  type Eta = "eta_001" | "eta_010";
  type Run = {
    run_id: string;
    config_id: string;
    method: Method;
    direction: "top" | "bottom";
    scope: Scope;
    candidate_run_id: string | null;
    big_recording_index: number | null;
    candidate_k: number;
    selection_k: number;
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
  type CurveSummaryRow = {
    config_id: string;
    method: Method | "random_stratified";
    direction: "top" | "bottom" | "random";
    scope: Scope;
    selection_eta: number;
    kernel_method: "multiplicative" | "additive" | null;
    w_interaction: number | null;
    recording_count: number;
    vendi_score_mean: number;
    vendi_score_q25: number;
    vendi_score_q75: number;
    mean_pairwise_cosine_mean: number;
    mean_pairwise_cosine_q25: number;
    mean_pairwise_cosine_q75: number;
    mean_recon_loss_mean: number;
    mean_recon_loss_q25: number;
    mean_recon_loss_q75: number;
  };

  const DEFAULT_ATLAS_URL =
    "https://huggingface.co/datasets/jonathan-lys/reve-atlas/resolve/main/data/display.parquet";
  const DEFAULT_SELECTIONS_URL =
    "https://huggingface.co/datasets/jonathan-lys/reve-atlas/resolve/main/data/selections.parquet";
  const DEFAULT_RUNS_URL =
    "https://huggingface.co/datasets/jonathan-lys/reve-atlas/resolve/main/data/selection_runs_web.parquet";
  const DEFAULT_CURVES_URL =
    "https://huggingface.co/datasets/jonathan-lys/reve-atlas/resolve/main/data/curve_summary.parquet";
  const atlasUrl =
    import.meta.env.VITE_ATLAS_ATLAS_URL ?? import.meta.env.VITE_ATLAS_DATA_URL ?? DEFAULT_ATLAS_URL;
  const selectionsUrl = import.meta.env.VITE_ATLAS_SELECTIONS_URL ?? DEFAULT_SELECTIONS_URL;
  const runsUrl = import.meta.env.VITE_ATLAS_RUNS_URL ?? DEFAULT_RUNS_URL;
  const curvesUrl = import.meta.env.VITE_ATLAS_CURVES_URL ?? DEFAULT_CURVES_URL;
  const datasetTokenUrl = new URL(`${import.meta.env.BASE_URL}dataset_token.json`, window.location.href).toString();
  const REQUIRED_COLUMNS = [
    "row_id", "window_id", "projection_x", "projection_y", "recon_loss",
    "big_recording_index", "session_index", "offset", "dataset", "modality",
    "sampling_rate", "n_channels",
  ].map((column) => `"${column}"`).join(", ");
  const DATASET_COLORS = [
    "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2",
    "#eeca3b", "#b279a2", "#ff9da6", "#9d755d", "#bab0ac",
  ];
  const POINT_LABEL_SQL = `(
    dataset || ' · ' || modality || ' · ' || n_channels::VARCHAR || ' ch · rec ' || big_recording_index::VARCHAR
  ) AS point_label`;
  // dataset_classes.task_modality/domain are comma-separated tag strings (a recording can carry
  // several tags at once). Keep them as plain strings on the table (embedding-atlas's default
  // per-column chart machinery doesn't handle LIST-typed columns well) and match individual tags
  // with list_contains(string_split(...)) inline in the quick-filter predicates below instead.
  const DOMAIN_TAGS = ["BCI", "Clinical", "Cognitive", "Evoked Response", "Internal State", "Sleep"];
  const TASK_MODALITY_TAGS = ["Auditory", "Clinical", "Cognitive", "Haptic", "Motor", "Passive", "Sleep", "Visual"];
  function tagPredicateItems(column: string, tags: string[]) {
    return tags.map((tag) => ({
      name: tag,
      predicate: `list_contains(string_split(${column}, ', '), '${tag}')`,
    }));
  }
  const DISPLAY_POINT_CAP = 2_000_000;
  const DISPLAY_SAMPLE_CAP = 1_760_000;
  const coordinator = new Coordinator();
  const webGpuAvailable = "gpu" in navigator;
  let dbConnector: Awaited<ReturnType<typeof wasmConnector>> | null = null;

  let realMode = false;
  let runs: Run[] = [];
  let activeRunId = "";
  let activeRun: Run | null = null;
  let validRuns: Run[] = [];
  let availableRecordings: string[] = [];
  let curveSummary: CurveSummaryRow[] = [];
  let selectedMethod: Method = "dpp";
  let selectedDirection: "top" | "bottom" = "top";
  let selectedEta: Eta = "eta_010";
  let selectedScope: Scope = "per_recording";
  let selectedRecording = "all";
  let selectedKernel: "multiplicative" | "additive" = "multiplicative";
  let viewMode: ViewMode = "all";
  let activeTableName = "";
  let refreshSerial = 0;
  let refreshChain: Promise<void> = Promise.resolve();
  // Both ranking and DPP config_ids carry a "eta_001"/"eta_010" tag marking the per-recording
  // candidate-pool size (1% or 10% of a recording's windows). Older/demo runs have neither tag
  // and are treated as eta-agnostic so they still show up.
  function runEta(run: Run): Eta | null {
    if (run.config_id.includes("eta_001")) return "eta_001";
    if (run.config_id.includes("eta_010")) return "eta_010";
    return null;
  }
  $: availableRecordings = [
    "all",
    ...new Set(
      runs.filter((run) => run.method === selectedMethod &&
        run.direction === selectedDirection &&
        run.scope === (selectedMethod === "dpp" ? "per_recording" : selectedScope) &&
        (selectedMethod === "ranking" || run.kernel_method === selectedKernel) &&
        (runEta(run) === selectedEta || runEta(run) === null) &&
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
      (selectedMethod === "ranking" || run.kernel_method === selectedKernel) &&
      (runEta(run) === selectedEta || runEta(run) === null);
  });
  $: activeRun = validRuns.find((run) => run.run_id === activeRunId) ?? validRuns[0] ?? null;
  function sameSelectionSweep(left: Run, right: Run): boolean {
    return left.method === right.method && left.config_id === right.config_id &&
      left.direction === right.direction && left.scope === right.scope &&
      left.kernel_method === right.kernel_method && left.w_interaction === right.w_interaction;
  }
  $: interactionRuns = (selectedRecording === "all" && selectedScope === "per_recording"
    ? validRuns.filter((run, index, all) => all.findIndex((candidate) => sameSelectionSweep(candidate, run)) === index)
    : validRuns
  ).slice().sort((a, b) => (a.w_interaction ?? -Infinity) - (b.w_interaction ?? -Infinity));
  $: sweepIndex = Math.max(0, interactionRuns.findIndex((run) => run.run_id === activeRunId));
  $: sweepLabel = interactionRuns[sweepIndex]
    ? (interactionRuns[sweepIndex].method === "ranking"
        ? `${interactionRuns[sweepIndex].actual_size} rows`
        : `w=${interactionRuns[sweepIndex].w_interaction} · ${interactionRuns[sweepIndex].actual_size} rows`)
    : "—";
  function onSweepInput(event: Event): void {
    const index = Number((event.currentTarget as HTMLInputElement).value);
    const run = interactionRuns[index];
    if (!run) return;
    activeRunId = run.run_id;
    void refreshActiveView(run);
  }
  function runsForActiveView(run: Run): Run[] {
    if (run.scope !== "per_recording" || selectedRecording !== "all") return [run];
    const matchingRuns = validRuns.filter((candidate) => sameSelectionSweep(candidate, run));
    return matchingRuns.length ? matchingRuns : [run];
  }
  $: activeViewRuns = activeRun ? runsForActiveView(activeRun) : [];
  $: activeViewCandidateK = activeViewRuns.reduce((total, run) => total + run.candidate_k, 0);
  $: activeViewSize = activeViewRuns.reduce((total, run) => total + run.actual_size, 0);
  type ChartPoint = { x: number; y: number; r: number; current: boolean; tooltip: string };
  type BaselinePoint = { x: number; y: number; xLo: number; xHi: number; yLo: number; yHi: number; tooltip: string };

  let showBaseline = true;
  // The operating curve aggregates every recording (via curve_summary.parquet)
  // rather than replaying one recording's noisy sweep, so it stays stable as
  // w varies and is comparable against the stratified baseline below.
  $: operatingCurve = activeRun?.method === "dpp"
    ? curveSummary
        .filter((row) => row.method === "dpp" && row.config_id === activeRun?.config_id)
        .sort((a, b) => (a.w_interaction ?? 0) - (b.w_interaction ?? 0))
    : [];
  $: strataBaseline = activeRun?.method === "dpp"
    ? curveSummary.find((row) => row.method === "random_stratified" && row.selection_eta === (selectedEta === "eta_010" ? 0.10 : 0.01)) ?? null
    : null;
  // The baseline sits far from the DPP curve on both metrics, so including it in
  // the axis bounds can squash the curve down to a sliver — showBaseline lets it
  // be excluded from both the plotted marker and the bounds it would otherwise stretch.
  $: effectiveBaseline = showBaseline ? strataBaseline : null;
  $: curvePoints = effectiveBaseline ? [...operatingCurve, effectiveBaseline] : operatingCurve;
  $: vendiMin = curvePoints.length ? Math.min(...curvePoints.map((row) => row.vendi_score_q25)) : 0;
  $: vendiMax = curvePoints.length ? Math.max(...curvePoints.map((row) => row.vendi_score_q75)) : 1;
  $: cosineMin = curvePoints.length ? Math.min(...curvePoints.map((row) => row.mean_pairwise_cosine_q25)) : 0;
  $: cosineMax = curvePoints.length ? Math.max(...curvePoints.map((row) => row.mean_pairwise_cosine_q75)) : 1;
  $: lossMin = curvePoints.length ? Math.min(...curvePoints.map((row) => row.mean_recon_loss_q25)) : 0;
  $: lossMax = curvePoints.length ? Math.max(...curvePoints.map((row) => row.mean_recon_loss_q75)) : 1;
  function scaleX(value: number, min: number, max: number): number {
    return 22 + ((value - min) / Math.max(max - min, 1e-9)) * 192;
  }
  function scaleY(value: number, min: number, max: number): number {
    return 108 - ((value - min) / Math.max(max - min, 1e-9)) * 96;
  }
  function buildChartPoints(
    rows: CurveSummaryRow[],
    xOf: (row: CurveSummaryRow) => number,
    xMin: number,
    xMax: number,
    xLabel: string,
  ): ChartPoint[] {
    return rows.map((row) => ({
      x: scaleX(xOf(row), xMin, xMax),
      y: scaleY(row.mean_recon_loss_mean, lossMin, lossMax),
      r: row.w_interaction === activeRun?.w_interaction ? 5 : 3,
      current: row.w_interaction === activeRun?.w_interaction,
      tooltip: `w=${row.w_interaction} · ${xLabel} ${xOf(row).toPrecision(4)} · loss ${row.mean_recon_loss_mean.toPrecision(4)} (mean of ${row.recording_count} recordings)`,
    }));
  }
  function buildBaselinePoint(
    baseline: CurveSummaryRow | null,
    xMeanOf: (row: CurveSummaryRow) => number,
    xQ25Of: (row: CurveSummaryRow) => number,
    xQ75Of: (row: CurveSummaryRow) => number,
    xMin: number,
    xMax: number,
    xLabel: string,
  ): BaselinePoint | null {
    if (!baseline) return null;
    return {
      x: scaleX(xMeanOf(baseline), xMin, xMax),
      y: scaleY(baseline.mean_recon_loss_mean, lossMin, lossMax),
      xLo: scaleX(xQ25Of(baseline), xMin, xMax),
      xHi: scaleX(xQ75Of(baseline), xMin, xMax),
      yLo: scaleY(baseline.mean_recon_loss_q25, lossMin, lossMax),
      yHi: scaleY(baseline.mean_recon_loss_q75, lossMin, lossMax),
      tooltip: `Stratified baseline (η=${(baseline.selection_eta * 100).toFixed(0)}%) · ${xLabel} ${xMeanOf(baseline).toPrecision(4)} · loss ${baseline.mean_recon_loss_mean.toPrecision(4)} (mean of ${baseline.recording_count} recordings, IQR whiskers shown)`,
    };
  }
  $: vendiChartPoints = buildChartPoints(operatingCurve, (row) => row.vendi_score_mean, vendiMin, vendiMax, "Vendi");
  $: vendiBaselinePoint = buildBaselinePoint(
    effectiveBaseline, (row) => row.vendi_score_mean, (row) => row.vendi_score_q25, (row) => row.vendi_score_q75,
    vendiMin, vendiMax, "Vendi",
  );
  $: cosineChartPoints = buildChartPoints(operatingCurve, (row) => row.mean_pairwise_cosine_mean, cosineMin, cosineMax, "cosine");
  $: cosineBaselinePoint = buildBaselinePoint(
    effectiveBaseline, (row) => row.mean_pairwise_cosine_mean, (row) => row.mean_pairwise_cosine_q25, (row) => row.mean_pairwise_cosine_q75,
    cosineMin, cosineMax, "cosine",
  );

  async function initialize(): Promise<void> {
    const wasm = await wasmConnector();
    dbConnector = wasm;
    coordinator.databaseConnector(wasm);
    await coordinator.exec(`CREATE OR REPLACE VIEW atlas_source AS SELECT * FROM read_parquet(${SQL.literal(atlasUrl)})`);
    // Small per-dataset lookup (task modality / domain), joined onto "dataset" — not worth
    // duplicating into the main parquet since it only varies per dataset, not per point.
    await coordinator.exec(`CREATE OR REPLACE TABLE dataset_classes AS SELECT * FROM read_json_auto(${SQL.literal(datasetTokenUrl)})`);
    realMode = Boolean(selectionsUrl && runsUrl);
    if (!realMode) {
      await coordinator.exec(`
        CREATE OR REPLACE TABLE display_points AS
        SELECT atlas_source.*, dataset_classes.task_modality, dataset_classes.domain, ${POINT_LABEL_SQL}
        FROM atlas_source
        LEFT JOIN dataset_classes USING (dataset)
        WHERE hash(row_id) % 100 < 8
        LIMIT ${DISPLAY_POINT_CAP}
      `);
      await coordinator.exec(`SELECT ${REQUIRED_COLUMNS} FROM display_points LIMIT 0`);
      return;
    }
    await coordinator.exec(`CREATE OR REPLACE TABLE selection_memberships AS SELECT * FROM read_parquet(${SQL.literal(selectionsUrl)})`);
    // Hugging Face serves LFS/Xet files through a redirect whose Content-Length
    // DuckDB-WASM can mistake for the Parquet size. The browser follows that
    // redirect correctly, so register the small web-specific table as a local
    // DuckDB file instead of asking DuckDB to open the remote URL itself.
    const runsResponse = await fetch(runsUrl, { cache: "no-store" });
    if (!runsResponse.ok) {
      throw new Error(`Could not load ${runsUrl}: HTTP ${runsResponse.status}`);
    }
    const runsFile = "selection_runs_web.parquet";
    const duckdb = await wasm.getDuckDB();
    await duckdb.registerFileBuffer(runsFile, new Uint8Array(await runsResponse.arrayBuffer()));
    await coordinator.exec(`
      CREATE OR REPLACE TABLE selection_runs AS
      SELECT
        run_id, config_id, method, direction, scope, candidate_run_id,
        big_recording_index, candidate_k, actual_size, kernel_method,
        w_interaction, vendi_score, log_det, mean_recon_loss,
        median_recon_loss, baseline_jaccard, adjacent_jaccard
      FROM read_parquet(${SQL.literal(runsFile)})
    `);
    await coordinator.exec(`CREATE OR REPLACE TABLE curve_summary_table AS SELECT * FROM read_parquet(${SQL.literal(curvesUrl)})`);
    curveSummary = (await coordinator.query("SELECT * FROM curve_summary_table", { type: "json" })) as CurveSummaryRow[];
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
      SELECT combined.*, dataset_classes.task_modality, dataset_classes.domain, ${POINT_LABEL_SQL} FROM (
        SELECT * FROM required_points
        UNION ALL
        SELECT * FROM sampled_points
      ) AS combined
      LEFT JOIN dataset_classes USING (dataset)
    `);
    await coordinator.exec(`SELECT ${REQUIRED_COLUMNS} FROM display_points LIMIT 0`);
    runs = (await coordinator.query(
      "SELECT * FROM selection_runs ORDER BY method, direction, scope, big_recording_index NULLS FIRST, run_id",
      { type: "json" },
    )) as Run[];
    if (!runs.length) throw new Error("The selection-runs table is empty.");
    const first = runs.find((run) => run.method === "dpp" && run.direction === "top" && run.scope === "per_recording" && runEta(run) === "eta_010")
      ?? runs.find((run) => run.method === "ranking" && run.direction === "top" && run.scope === "global")
      ?? runs[0];
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
      // embedding-atlas caches per-table category/legend metadata by table name, so reusing
      // "active_points" across filter changes can serve stale colors/labels (or blank points)
      // for the rebuilt table. Give every refresh a fresh name instead.
      const nextTableName = `active_points_${requestId}`;
      await coordinator.exec(`
      CREATE OR REPLACE TABLE ${nextTableName} AS
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
      if (requestId !== refreshSerial) {
        await coordinator.exec(`DROP TABLE IF EXISTS ${nextTableName}`);
        return;
      }
      const previousTableName = activeTableName;
      activeTableName = nextTableName;
      if (previousTableName) await coordinator.exec(`DROP TABLE IF EXISTS ${previousTableName}`);
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

  async function exportSelection(
    predicate: string | null,
    format: "json" | "jsonl" | "csv" | "parquet",
  ): Promise<void> {
    if (!dbConnector) return;
    const table = realMode ? activeTableName : "display_points";
    const where = predicate ? `WHERE ${predicate}` : "";
    const copyOptions = format === "csv" ? "(FORMAT CSV, HEADER)" :
      format === "parquet" ? "(FORMAT PARQUET)" :
      format === "jsonl" ? "(FORMAT JSON, ARRAY false)" : "(FORMAT JSON, ARRAY true)";
    const outputPath = `export.${format === "jsonl" ? "jsonl" : format}`;
    const db = await dbConnector.getDuckDB();
    const connection = await dbConnector.getConnection();
    await connection.query(`COPY (SELECT * FROM ${table} ${where}) TO '${outputPath}' ${copyOptions}`);
    const buffer = await db.copyFileToBuffer(outputPath);
    const bytes = new Uint8Array(buffer.byteLength);
    bytes.set(buffer);
    const mimeType = format === "csv" ? "text/csv" : format === "parquet" ? "application/octet-stream" : "application/json";
    const blob = new Blob([bytes], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `reve-atlas-selection.${outputPath.split(".").pop()}`;
    link.click();
    URL.revokeObjectURL(url);
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
      <aside class="side-rail" aria-label="Atlas controls and active run summary">
        <div class="panel-section">
          <p class="eyebrow">REVE explorer</p>
          <h1>Embedding Atlas</h1>
          <p class="rail-copy">Switch among precomputed rankings and diversity selections.</p>
          <p class="display-note">Display cap: about 2M points; all selection members are retained. Per-recording views union matching intra-index runs.</p>
          <div class="control-stack">
            <label>Method<select bind:value={selectedMethod} onchange={onFilterChange}><option value="dpp">DPP selection</option><option value="ranking">Ranking</option></select></label>
            <label>Direction<select bind:value={selectedDirection} onchange={onFilterChange}><option value="top">Top loss</option><option value="bottom">Bottom loss</option></select></label>
            <label>Candidate pool<select bind:value={selectedEta} onchange={onFilterChange}><option value="eta_010">10% per recording</option><option value="eta_001">1% per recording</option></select></label>
            <label>Scope<select bind:value={selectedScope} onchange={onFilterChange}><option value="global">Global</option><option value="per_recording">Per recording</option></select></label>
            <label>Recording<select bind:value={selectedRecording} onchange={onFilterChange} disabled={selectedScope === "global"}>{#each availableRecordings as recording}<option value={recording}>{recording === "all" ? "All indexed recordings" : recording}</option>{/each}</select></label>
            <label>Kernel<select bind:value={selectedKernel} onchange={onFilterChange} disabled={selectedMethod === "ranking"}><option value="multiplicative">Multiplicative · utility ↑</option><option value="additive">Additive · diversity ↑</option></select></label>
            <label>Selection sweep
              <input type="range" min="0" max={Math.max(0, interactionRuns.length - 1)} step="1"
                value={sweepIndex} oninput={onSweepInput} disabled={interactionRuns.length <= 1} />
              <span class="sweep-value">{sweepLabel}</span>
            </label>
            <label>View<select bind:value={viewMode} onchange={() => refreshActiveView(activeRun)}><option value="all">All atlas points</option><option value="candidate">Candidate pool</option><option value="selected">Selected rows only</option></select></label>
          </div>
        </div>
        {#if activeRun}
          <div class="panel-section">
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
            {#snippet operatingCurveChart(title: string, points: ChartPoint[], baselinePoint: BaselinePoint | null, xAxisLabel: string, note: string)}
              <div class="tradeoff">
                <h4>{title} / loss</h4>
                <svg viewBox="0 0 220 130" role="img" aria-label="{title} versus mean reconstruction loss: DPP operating curve versus stratified baseline, aggregated across recordings">
                  <line x1="22" y1="8" x2="22" y2="108" />
                  <line x1="22" y1="108" x2="214" y2="108" />
                  {#if baselinePoint}
                    <line class="baseline-guide" x1="22" y1={baselinePoint.y} x2="214" y2={baselinePoint.y} />
                    <line class="baseline-guide" x1={baselinePoint.x} y1="8" x2={baselinePoint.x} y2="108" />
                  {/if}
                  <polyline class="curve-line" points={points.map((point) => `${point.x},${point.y}`).join(" ")} />
                  {#each points as point}
                    <circle class="curve-point" class:current={point.current} cx={point.x} cy={point.y} r={point.r}><title>{point.tooltip}</title></circle>
                  {/each}
                  {#if baselinePoint}
                    <line class="baseline-whisker" x1={baselinePoint.xLo} y1={baselinePoint.y} x2={baselinePoint.xHi} y2={baselinePoint.y} />
                    <line class="baseline-whisker" x1={baselinePoint.x} y1={baselinePoint.yLo} x2={baselinePoint.x} y2={baselinePoint.yHi} />
                    <rect
                      class="baseline-marker"
                      x={baselinePoint.x - 4}
                      y={baselinePoint.y - 4}
                      width="8" height="8"
                      transform={`rotate(45 ${baselinePoint.x} ${baselinePoint.y})`}
                    ><title>{baselinePoint.tooltip}</title></rect>
                  {/if}
                </svg>
                <div class="axis-labels"><span>{xAxisLabel}</span><span>loss ↑</span></div>
                <ul class="legend">
                  <li><span class="swatch curve"></span>DPP sweep ({activeRun.kernel_method})</li>
                  {#if baselinePoint}<li><span class="swatch baseline"></span>Stratified baseline</li>{/if}
                </ul>
                <small>{note}</small>
              </div>
            {/snippet}
            {#if operatingCurve.length > 1}
              <div class="tradeoff-group">
                <div class="tradeoff-header">
                  <h3>Operating curves</h3>
                  {#if strataBaseline}
                    <label class="baseline-toggle"><input type="checkbox" bind:checked={showBaseline} /> Show baseline</label>
                  {/if}
                </div>
                {@render operatingCurveChart(
                  "Vendi diversity",
                  vendiChartPoints,
                  vendiBaselinePoint,
                  "Vendi diversity →",
                  activeRun.kernel_method === "multiplicative" ? "Higher w emphasizes ranking utility." : "Higher w emphasizes interaction/diversity.",
                )}
                {@render operatingCurveChart(
                  "Cosine similarity",
                  cosineChartPoints,
                  cosineBaselinePoint,
                  "← more diverse · cosine similarity · less diverse →",
                  "Mean pairwise cosine of selected rows; lower means more diverse.",
                )}
                {#if strataBaseline}<small class="baseline-note">Baseline whiskers show the interquartile range across recordings.</small>{/if}
              </div>
            {/if}
          </div>
        {/if}
      </aside>
    {/if}
    <section class="atlas-shell">
      {#key realMode ? activeTableName : "demo"}
        <EmbeddingAtlas
          {coordinator}
          data={{ table: realMode ? activeTableName : "display_points", id: "row_id", text: "point_label", projection: { x: "projection_x", y: "projection_y" } }}
          defaultChartsConfig={{
            include: realMode
              ? ["recon_loss", "dataset", "n_channels", "modality", "task_modality", "domain", "big_recording_index", "selected", "candidate"]
              : ["recon_loss", "dataset", "n_channels", "modality", "task_modality", "domain"],
            override: {
              modality: {
                type: "predicates",
                title: "Modality",
                items: [
                  { name: "EEG only", predicate: "modality = 'EEG'" },
                  { name: "MEG only", predicate: "modality = 'MEG'" },
                ],
              },
              task_modality: { type: "predicates", title: "Task modality", items: tagPredicateItems("task_modality", TASK_MODALITY_TAGS) },
              domain: { type: "predicates", title: "Domain", items: tagPredicateItems("domain", DOMAIN_TAGS) },
            },
            embedding: { data: { x: "projection_x", y: "projection_y", text: "point_label", category: "dataset" } },
            table: false,
          }}
          chartTheme={{ categoryColors: DATASET_COLORS }}
          embeddingViewConfig={{ downsampleMaxPoints: DISPLAY_POINT_CAP, pointSize: 1.5 }}
          onExportSelection={exportSelection}
        />
      {/key}
    </section>
  </main>
{:catch error}
  <main class="status error" role="alert"><div class="status-card"><p class="eyebrow">Dataset unavailable</p><h1>REVE Embedding Atlas could not start</h1><p>{describeError(error)}</p><button type="button" onclick={() => window.location.reload()}>Retry</button></div></main>
{/await}

<style>
  :root {
    --bg: #f5f7fb;
    --panel-bg: rgb(255 255 255 / 96%);
    --border: #dfe5f0;
    --text-muted: #526078;
    --text-faint: #7c879b;
    --text-soft: #69758b;
    --heading: #172033;
    --eyebrow: #c2415a;
    --input-bg: #fff;
    --input-text: #172033;
    --input-border: #d4dbea;
    --input-disabled-bg: #f3f5f9;
    --input-disabled-text: #8b95a8;
    --hero-bg: #eef0ff;
    --hero-text: #202d91;
    --accent: #5665ff;
    --accent-strong: #4c5cf4;
    --curve-baseline: #008300;
    --shadow: rgb(22 32 51 / 12%);
    --status-card-bg: rgb(255 255 255 / 92%);
    --run-id-text: #8792a7;
    --spinner-track: #dce2ee;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #12141c;
      --panel-bg: rgb(24 27 38 / 96%);
      --border: #2a2f3f;
      --text-muted: #a7b0c4;
      --text-faint: #7d879c;
      --text-soft: #929cb2;
      --heading: #eef1f8;
      --eyebrow: #ff7a92;
      --input-bg: #1c1f2c;
      --input-text: #eef1f8;
      --input-border: #333952;
      --input-disabled-bg: #1a1d28;
      --input-disabled-text: #5c6479;
      --hero-bg: #1e2350;
      --hero-text: #b9c1ff;
      --accent: #7c86ff;
      --accent-strong: #6672ff;
      --shadow: rgb(0 0 0 / 45%);
      --status-card-bg: rgb(24 27 38 / 92%);
      --run-id-text: #838da3;
      --spinner-track: #333952;
    }
  }
  .atlas-app { display: grid; grid-template-columns: 20rem minmax(0, 1fr); min-width: 0; min-height: 0; width: 100%; height: 100%; overflow: hidden; background: var(--bg); }
  .atlas-app:not(.real-mode) { display: block; }
  .atlas-shell { min-width: 0; min-height: 0; width: 100%; height: 100%; overflow: hidden; position: relative; }
  .side-rail { z-index: 2; box-sizing: border-box; display: flex; flex-direction: column; min-width: 0; min-height: 0; overflow: auto; border-right: 1px solid var(--border); background: var(--panel-bg); }
  .panel-section { box-sizing: border-box; padding: 1.25rem 1rem; }
  .panel-section + .panel-section { border-top: 1px solid var(--border); }
  h1 { margin: 0 0 0.55rem; color: var(--heading); font-size: 1.45rem; letter-spacing: -0.04em; }
  h2 { margin: 0.1rem 0 0.3rem; color: var(--heading); font-size: 1.15rem; }
  h3 { margin: 1.25rem 0 0.25rem; color: var(--text-muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }
  p { margin: 0; color: var(--text-muted); line-height: 1.45; }
  .rail-copy { margin-bottom: 0.35rem; font-size: 0.78rem; }
  .display-note { margin-bottom: 1.25rem; color: var(--text-faint); font-size: 0.68rem; }
  .eyebrow { margin-bottom: 0.45rem; color: var(--eyebrow); font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; }
  .control-stack { display: grid; gap: 0.75rem; }
  label { display: grid; gap: 0.25rem; color: var(--text-muted); font-size: 0.68rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }
  select { width: 100%; min-width: 0; box-sizing: border-box; padding: 0.45rem 0.4rem; border: 1px solid var(--input-border); border-radius: 0.35rem; color: var(--input-text); background: var(--input-bg); font-size: 0.75rem; text-transform: none; }
  select:disabled { color: var(--input-disabled-text); background: var(--input-disabled-bg); }
  input[type="range"] { width: 100%; accent-color: var(--accent); }
  input[type="range"]:disabled { opacity: 0.5; }
  .sweep-value { color: var(--text-soft); font-size: 0.65rem; font-weight: 400; letter-spacing: normal; text-transform: none; }
  .run-id { overflow: hidden; margin-bottom: 0.45rem; color: var(--run-id-text); font: 0.6rem ui-monospace, monospace; text-overflow: ellipsis; white-space: nowrap; }
  .view-note { margin-bottom: 1rem; color: var(--text-soft); font-size: 0.68rem; }
  .metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem 0.55rem; }
  .metric { display: grid; gap: 0.1rem; }
  .metric strong { color: var(--hero-text); font-size: 0.92rem; }
  .metric span, small, .axis-labels { color: var(--text-soft); font-size: 0.65rem; }
  .metric.hero { grid-column: 1 / -1; padding: 0.7rem; border-radius: 0.45rem; background: var(--hero-bg); }
  .metric.hero strong { font-size: 1.5rem; }
  .tradeoff-group { margin-top: 1.25rem; }
  .tradeoff-header { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 0.5rem; }
  .tradeoff-header h3 { margin: 0; }
  .baseline-toggle { display: flex; align-items: center; gap: 0.3rem; color: var(--text-soft); font-size: 0.65rem; font-weight: 400; letter-spacing: normal; text-transform: none; }
  .baseline-toggle input { accent-color: var(--curve-baseline); }
  .baseline-note { display: block; margin-top: 0.35rem; }
  .tradeoff { margin-top: 0.75rem; }
  .tradeoff h4 { margin: 0 0 0.3rem; color: var(--text-muted); font-size: 0.68rem; font-weight: 600; text-transform: none; }
  .tradeoff svg { display: block; width: 100%; height: 8rem; overflow: visible; }
  .tradeoff line { stroke: var(--input-border); stroke-width: 1; }
  .tradeoff line.baseline-guide { stroke: var(--curve-baseline); stroke-width: 1; stroke-dasharray: 2 2; opacity: 0.55; }
  .tradeoff line.baseline-whisker { stroke: var(--curve-baseline); stroke-width: 1.25; }
  .tradeoff polyline.curve-line { fill: none; stroke: var(--accent); stroke-width: 1.5; }
  .tradeoff circle.curve-point { fill: var(--accent); }
  .tradeoff circle.curve-point.current { fill: var(--eyebrow); stroke: var(--panel-bg); stroke-width: 1.5; }
  .tradeoff rect.baseline-marker { fill: var(--curve-baseline); stroke: var(--panel-bg); stroke-width: 1; }
  .tradeoff .legend { display: flex; flex-wrap: wrap; gap: 0.6rem; margin: 0.5rem 0 0.35rem; padding: 0; list-style: none; color: var(--text-muted); font-size: 0.65rem; }
  .tradeoff .legend li { display: flex; align-items: center; gap: 0.3rem; }
  .tradeoff .swatch { display: inline-block; width: 0.55rem; height: 0.55rem; border-radius: 0.15rem; }
  .tradeoff .swatch.curve { background: var(--accent); }
  .tradeoff .swatch.baseline { background: var(--curve-baseline); transform: rotate(45deg); }
  .axis-labels { display: flex; justify-content: space-between; }
  .status { box-sizing: border-box; display: grid; width: 100%; height: 100%; place-items: center; padding: 2rem; background: radial-gradient(circle at 20% 20%, rgb(86 101 255 / 14%), transparent 32rem), radial-gradient(circle at 80% 70%, rgb(34 197 175 / 12%), transparent 28rem), var(--bg); }
  .status-card { width: min(42rem, 100%); padding: 2.5rem; border: 1px solid var(--border); border-radius: 1rem; background: var(--status-card-bg); box-shadow: 0 1.25rem 4rem var(--shadow); }
  .spinner { width: 1.75rem; height: 1.75rem; margin-bottom: 1.25rem; border: 3px solid var(--spinner-track); border-top-color: var(--accent); border-radius: 999px; animation: spin 0.8s linear infinite; }
  button { margin-top: 1.5rem; padding: 0.65rem 1rem; border: 0; border-radius: 0.5rem; color: white; background: var(--accent-strong); cursor: pointer; }
  @media (max-width: 1050px) { .atlas-app.real-mode { grid-template-columns: 16rem minmax(0, 1fr); } }
  @media (max-width: 700px) { .atlas-app.real-mode { display: block; overflow: auto; } .side-rail { border: 0; border-bottom: 1px solid var(--border); } .atlas-shell { height: 70vh; min-height: 32rem; } }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
