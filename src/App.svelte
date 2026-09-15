<script lang="ts">
  import { Coordinator, wasmConnector } from "@uwdata/mosaic-core";
  import * as SQL from "@uwdata/mosaic-sql";
  import { EmbeddingAtlas } from "embedding-atlas/svelte";

  type Method = "ranking" | "dpp";
  type Scope = "global" | "per_recording";
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
  const CHANNEL_PREDICATE_ITEMS = [
    { name: "19 ch", predicate: "n_channels = 19" },
    { name: "21 ch", predicate: "n_channels = 21" },
    { name: "22 ch", predicate: "n_channels = 22" },
    { name: "Others LO (≤ 64 ch)", predicate: "n_channels <= 64 AND n_channels NOT IN (19, 21, 22)" },
    { name: "125 ch", predicate: "n_channels = 125" },
    { name: "128 ch", predicate: "n_channels = 128" },
    { name: "129 ch", predicate: "n_channels = 129" },
    { name: "Others HI (> 64 ch)", predicate: "n_channels > 64 AND n_channels NOT IN (125, 128, 129)" },
  ];
  const CHANNEL_GROUP_SQL = `(
    CASE
      WHEN n_channels = 19 THEN '19 ch'
      WHEN n_channels = 21 THEN '21 ch'
      WHEN n_channels = 22 THEN '22 ch'
      WHEN n_channels = 125 THEN '125 ch'
      WHEN n_channels = 128 THEN '128 ch'
      WHEN n_channels = 129 THEN '129 ch'
      WHEN n_channels <= 64 THEN 'Others LO (≤ 64 ch)'
      ELSE 'Others HI (> 64 ch)'
    END
  ) AS channel_group`;
  // Every dataset_classes column other than "dataset" is a label column discovered at load time, so
  // adding a label only means adding a key to dataset_token.json. Multi-tag columns are comma-separated
  // strings (a recording can carry several tags at once): keep them as plain strings on the table
  // (embedding-atlas's default per-column chart machinery doesn't handle LIST-typed columns well) and
  // match individual tags with list_contains(string_split(...)) in the quick-filter predicates instead.
  const TAG_SEPARATOR = ", ";
  type PredicateChart = { type: "predicates"; title: string; items: { name: string; predicate: string }[] };
  let labelColumns: string[] = [];
  let labelCharts: Record<string, PredicateChart> = {};
  function quoteIdent(name: string): string {
    return `"${name.replace(/"/g, '""')}"`;
  }
  function chartTitle(column: string): string {
    const words = column.replace(/_/g, " ");
    return words.charAt(0).toUpperCase() + words.slice(1);
  }
  const DPP_FILTERS = [
    { key: "eta_001_multiplicative", eta: 0.01, etaLabel: "1%", kernel: "multiplicative", kernelLabel: "Multiplicative" },
    { key: "eta_001_additive", eta: 0.01, etaLabel: "1%", kernel: "additive", kernelLabel: "Additive" },
    { key: "eta_010_multiplicative", eta: 0.10, etaLabel: "10%", kernel: "multiplicative", kernelLabel: "Multiplicative" },
    { key: "eta_010_additive", eta: 0.10, etaLabel: "10%", kernel: "additive", kernelLabel: "Additive" },
  ] as const;
  type DppFilter = (typeof DPP_FILTERS)[number];
  function dppColumn(direction: "top" | "bottom", filter: DppFilter): string {
    return `dpp_${direction}_${filter.key}`;
  }
  function tagPredicateItems(column: string, tags: string[]) {
    return tags.map((tag) => ({
      name: tag,
      predicate: `list_contains(string_split(${quoteIdent(column)}, ${SQL.literal(TAG_SEPARATOR)}), ${SQL.literal(tag)})`,
    }));
  }
  const DISPLAY_POINT_CAP = 2_000_000;
  const DISPLAY_SAMPLE_CAP = 1_760_000;
  const coordinator = new Coordinator();
  const webGpuAvailable = "gpu" in navigator;
  let dbConnector: Awaited<ReturnType<typeof wasmConnector>> | null = null;

  let realMode = false;
  let runs: Run[] = [];
  let curveSummary: CurveSummaryRow[] = [];
  let railTab: "explore" | "curves" = "explore";
  let activeTableName = "";
  let atlasDppMode: "all" | "selected" = "all";
  let atlasDppDirection: "top" | "bottom" = "bottom";
  let atlasDppEta: Eta = "eta_010";
  let atlasDppKernel: "multiplicative" | "additive" = "multiplicative";
  let atlasWIndex = 0;
  const atlasDppFilterSource = { reset: () => atlasDppMode = "all" };
  $: atlasWValues = [...new Set(
    runs.filter((run) => run.method === "dpp" && run.direction === atlasDppDirection &&
      run.selection_eta === (atlasDppEta === "eta_010" ? 0.10 : 0.01) && run.kernel_method === atlasDppKernel)
      .map((run) => run.w_interaction)
      .filter((value): value is number => value !== null),
  )].sort((a, b) => a - b);
  $: atlasWValue = atlasWValues[Math.min(atlasWIndex, Math.max(0, atlasWValues.length - 1))] ?? null;

  async function applyAtlasDppFilter(resetWeight = false): Promise<void> {
    if (resetWeight) atlasWIndex = 0;
    let group = [...coordinator.filterGroups.values()].sort((a, b) => b.clients.size - a.clients.size)[0];
    for (let attempt = 0; !group && attempt < 20; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 50));
      group = [...coordinator.filterGroups.values()].sort((a, b) => b.clients.size - a.clients.size)[0];
    }
    if (!group) return;
    const filter = DPP_FILTERS.find((item) =>
      item.eta === (atlasDppEta === "eta_010" ? 0.10 : 0.01) && item.kernel === atlasDppKernel);
    const weightIndex = resetWeight ? 0 : Math.min(atlasWIndex, Math.max(0, atlasWValues.length - 1));
    const value = atlasWValues[weightIndex] ?? null;
    group.selection.update({
      source: atlasDppFilterSource,
      value: atlasDppMode === "selected" ? { direction: atlasDppDirection, eta: atlasDppEta, kernel: atlasDppKernel, w: value } : null,
      predicate: atlasDppMode === "selected" && filter && value !== null
        ? SQL.listContains(dppColumn(atlasDppDirection, filter), value)
        : null,
    });
  }
  type ChartPoint = {
    x: number;
    y: number;
    r: number;
    current: boolean;
    tooltip: string;
    w: number | null;
    kernel: "multiplicative" | "additive" | null;
  };
  type BaselinePoint = { x: number; y: number; xLo: number; xHi: number; yLo: number; yHi: number; tooltip: string };

  let showBaseline = true;
  let curveDirection: "top" | "bottom" = "bottom";
  let curveEta: Eta = "eta_010";
  let curveKernel: "multiplicative" | "additive" = "multiplicative";
  // The operating curves aggregate every recording (via curve_summary.parquet)
  // rather than replaying one recording's noisy sweep, so they stay stable as
  // w varies and are comparable against the stratified baseline below.
  $: curveMultiplicative = curveSummary
    .filter((row) => row.method === "dpp" && row.direction === curveDirection &&
      row.selection_eta === (curveEta === "eta_010" ? 0.10 : 0.01) && row.kernel_method === "multiplicative")
    .sort((a, b) => (a.w_interaction ?? 0) - (b.w_interaction ?? 0));
  $: curveAdditive = curveSummary
    .filter((row) => row.method === "dpp" && row.direction === curveDirection &&
      row.selection_eta === (curveEta === "eta_010" ? 0.10 : 0.01) && row.kernel_method === "additive")
    .sort((a, b) => (a.w_interaction ?? 0) - (b.w_interaction ?? 0));
  $: strataBaseline = curveSummary.find((row) =>
    row.method === "random_stratified" && row.selection_eta === (curveEta === "eta_010" ? 0.10 : 0.01)) ?? null;
  // The baseline sits far from the DPP curve on both metrics, so including it in
  // the axis bounds can squash the curve down to a sliver — showBaseline lets it
  // be excluded from both the plotted marker and the bounds it would otherwise stretch.
  $: effectiveBaseline = showBaseline ? strataBaseline : null;
  $: curvePoints = effectiveBaseline
    ? [...curveMultiplicative, ...curveAdditive, effectiveBaseline]
    : [...curveMultiplicative, ...curveAdditive];
  $: vendiMin = curvePoints.length ? Math.min(...curvePoints.map((row) => row.vendi_score_q25 ?? row.vendi_score_mean)) : 0;
  $: vendiMax = curvePoints.length ? Math.max(...curvePoints.map((row) => row.vendi_score_q75 ?? row.vendi_score_mean)) : 1;
  $: cosineMin = curvePoints.length ? Math.min(...curvePoints.map((row) => row.mean_pairwise_cosine_q25 ?? row.mean_pairwise_cosine_mean)) : 0;
  $: cosineMax = curvePoints.length ? Math.max(...curvePoints.map((row) => row.mean_pairwise_cosine_q75 ?? row.mean_pairwise_cosine_mean)) : 1;
  $: lossMin = curvePoints.length ? Math.min(...curvePoints.map((row) => row.mean_recon_loss_q25 ?? row.mean_recon_loss_mean)) : 0;
  $: lossMax = curvePoints.length ? Math.max(...curvePoints.map((row) => row.mean_recon_loss_q75 ?? row.mean_recon_loss_mean)) : 1;
  const CHART_LEFT = 34;
  const CHART_RIGHT = 406;
  const CHART_TOP = 14;
  const CHART_BOTTOM = 238;
  function scaleX(value: number, min: number, max: number): number {
    return CHART_LEFT + ((value - min) / Math.max(max - min, 1e-9)) * (CHART_RIGHT - CHART_LEFT);
  }
  function scaleY(value: number, min: number, max: number): number {
    return CHART_BOTTOM - ((value - min) / Math.max(max - min, 1e-9)) * (CHART_BOTTOM - CHART_TOP);
  }
  function buildChartPoints(
    rows: CurveSummaryRow[],
    xOf: (row: CurveSummaryRow) => number,
    xMin: number,
    xMax: number,
    xLabel: string,
    yMin = lossMin,
    yMax = lossMax,
    currentWeight: number | null = null,
  ): ChartPoint[] {
    return rows.map((row) => ({
      x: scaleX(xOf(row), xMin, xMax),
      y: scaleY(row.mean_recon_loss_mean, yMin, yMax),
      r: row.w_interaction === currentWeight ? 6 : 4,
      current: row.w_interaction === currentWeight,
      w: row.w_interaction,
      kernel: (row.kernel_method === "multiplicative" || row.kernel_method === "additive") ? row.kernel_method : null,
      tooltip: `${row.kernel_method ? `[${row.kernel_method}] ` : ""}w=${row.w_interaction} · ${xLabel} ${xOf(row).toPrecision(4)} · loss ${row.mean_recon_loss_mean.toPrecision(4)} (mean of ${row.recording_count} recordings)`,
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
  $: curveActive = curveKernel === "multiplicative" ? curveMultiplicative : curveAdditive;
  $: curveShaded = curveKernel === "multiplicative" ? curveAdditive : curveMultiplicative;

  $: vendiActivePoints = buildChartPoints(curveActive, (row) => row.vendi_score_mean, vendiMin, vendiMax, "Vendi");
  $: vendiShadedPoints = buildChartPoints(curveShaded, (row) => row.vendi_score_mean, vendiMin, vendiMax, "Vendi");
  $: vendiBaselinePoint = buildBaselinePoint(
    effectiveBaseline, (row) => row.vendi_score_mean, (row) => row.vendi_score_q25, (row) => row.vendi_score_q75,
    vendiMin, vendiMax, "Vendi",
  );
  $: cosineActivePoints = buildChartPoints(curveActive, (row) => row.mean_pairwise_cosine_mean, cosineMin, cosineMax, "cosine");
  $: cosineShadedPoints = buildChartPoints(curveShaded, (row) => row.mean_pairwise_cosine_mean, cosineMin, cosineMax, "cosine");
  $: cosineBaselinePoint = buildBaselinePoint(
    effectiveBaseline, (row) => row.mean_pairwise_cosine_mean, (row) => row.mean_pairwise_cosine_q25, (row) => row.mean_pairwise_cosine_q75,
    cosineMin, cosineMax, "cosine",
  );

  $: atlasMultiplicativeCurve = curveSummary
    .filter((row) => row.method === "dpp" && row.direction === atlasDppDirection &&
      row.selection_eta === (atlasDppEta === "eta_010" ? 0.10 : 0.01) && row.kernel_method === "multiplicative")
    .sort((a, b) => (a.w_interaction ?? 0) - (b.w_interaction ?? 0));
  $: atlasAdditiveCurve = curveSummary
    .filter((row) => row.method === "dpp" && row.direction === atlasDppDirection &&
      row.selection_eta === (atlasDppEta === "eta_010" ? 0.10 : 0.01) && row.kernel_method === "additive")
    .sort((a, b) => (a.w_interaction ?? 0) - (b.w_interaction ?? 0));
  $: atlasCombinedCurves = [...atlasMultiplicativeCurve, ...atlasAdditiveCurve];
  $: atlasVendiMin = atlasCombinedCurves.length ? Math.min(...atlasCombinedCurves.map((row) => row.vendi_score_mean)) : 0;
  $: atlasVendiMax = atlasCombinedCurves.length ? Math.max(...atlasCombinedCurves.map((row) => row.vendi_score_mean)) : 1;
  $: atlasLossMin = atlasCombinedCurves.length ? Math.min(...atlasCombinedCurves.map((row) => row.mean_recon_loss_mean)) : 0;
  $: atlasLossMax = atlasCombinedCurves.length ? Math.max(...atlasCombinedCurves.map((row) => row.mean_recon_loss_mean)) : 1;
  $: atlasMultiplicativePoints = buildChartPoints(
    atlasMultiplicativeCurve, (row) => row.vendi_score_mean, atlasVendiMin, atlasVendiMax, "Vendi",
    atlasLossMin, atlasLossMax, atlasDppKernel === "multiplicative" ? atlasWValue : null,
  );
  $: atlasAdditivePoints = buildChartPoints(
    atlasAdditiveCurve, (row) => row.vendi_score_mean, atlasVendiMin, atlasVendiMax, "Vendi",
    atlasLossMin, atlasLossMax, atlasDppKernel === "additive" ? atlasWValue : null,
  );
  $: atlasActivePoints = atlasDppKernel === "multiplicative" ? atlasMultiplicativePoints : atlasAdditivePoints;
  $: atlasShadedPoints = atlasDppKernel === "multiplicative" ? atlasAdditivePoints : atlasMultiplicativePoints;

  function selectAtlasPoint(point: ChartPoint): void {
    if (point.kernel && point.kernel !== atlasDppKernel) {
      atlasDppKernel = point.kernel;
      const targetW = point.w;
      const kernelRuns = runs.filter((run) =>
        run.method === "dpp" &&
        run.direction === atlasDppDirection &&
        run.selection_eta === (atlasDppEta === "eta_010" ? 0.10 : 0.01) &&
        run.kernel_method === point.kernel,
      );
      const wValues = [...new Set(kernelRuns.map((r) => r.w_interaction).filter((v): v is number => v !== null))].sort((a, b) => a - b);
      if (targetW !== null) {
        const idx = wValues.indexOf(targetW);
        atlasWIndex = idx >= 0 ? idx : 0;
      } else {
        atlasWIndex = 0;
      }
      applyAtlasDppFilter();
    } else {
      if (point.w !== null) {
        const idx = atlasWValues.indexOf(point.w);
        if (idx >= 0) atlasWIndex = idx;
      }
      applyAtlasDppFilter();
    }
  }

  async function describeColumns(relation: string): Promise<{ column_name: string; column_type: string }[]> {
    return (await coordinator.query(`SELECT column_name, column_type FROM (DESCRIBE ${relation})`, {
      type: "json",
    })) as { column_name: string; column_type: string }[];
  }

  // Reads dataset_token.json, keeps every column beside "dataset" as a label column, and derives the
  // quick-filter tag lists from the file itself so new labels need no matching code change here.
  async function loadDatasetClasses(): Promise<void> {
    await coordinator.exec(
      `CREATE OR REPLACE TABLE dataset_classes_raw AS SELECT * FROM read_json_auto(${SQL.literal(datasetTokenUrl)})`,
    );
    const atlasColumns = new Set((await describeColumns("atlas_source")).map((column) => column.column_name));
    const candidates = (await describeColumns("dataset_classes_raw")).filter(
      (column) => column.column_name !== "dataset",
    );
    const shadowed = candidates.filter((column) => atlasColumns.has(column.column_name));
    if (shadowed.length) {
      console.warn(
        `Ignoring dataset_token.json columns that already exist on the atlas table: ${shadowed.map((column) => column.column_name).join(", ")}`,
      );
    }
    const labels = candidates.filter((column) => !atlasColumns.has(column.column_name));
    // Canonicalize hand-edited separators ("A,B" or "A ,  B" -> "A, B") so tag extraction and the
    // list_contains predicates agree on where one tag ends and the next begins.
    const projected = labels.map((column) =>
      column.column_type === "VARCHAR"
        ? `regexp_replace(trim(${quoteIdent(column.column_name)}), '\\s*,\\s*', ${SQL.literal(TAG_SEPARATOR)}, 'g') AS ${quoteIdent(column.column_name)}`
        : quoteIdent(column.column_name),
    );
    await coordinator.exec(`
      CREATE OR REPLACE TABLE dataset_classes AS
      SELECT dataset${projected.length ? `, ${projected.join(", ")}` : ""} FROM dataset_classes_raw
    `);
    labelColumns = labels.map((column) => column.column_name);
    const charts: Record<string, PredicateChart> = {};
    for (const column of labels.filter((item) => item.column_type === "VARCHAR")) {
      const name = quoteIdent(column.column_name);
      const split = `string_split(${name}, ${SQL.literal(TAG_SEPARATOR)})`;
      const tagRows = (await coordinator.query(
        `
        SELECT tag, max(tag_count) AS tag_count FROM (
          SELECT unnest(${split}) AS tag, len(${split}) AS tag_count
          FROM dataset_classes WHERE ${name} IS NOT NULL AND ${name} <> ''
        ) GROUP BY tag ORDER BY tag
      `,
        { type: "json" },
      )) as { tag: string; tag_count: number }[];
      // Single-valued columns keep embedding-atlas's own categorical chart; only genuinely multi-tag
      // columns need the list_contains quick filters.
      if (tagRows.some((row) => Number(row.tag_count) > 1)) {
        charts[column.column_name] = {
          type: "predicates",
          title: chartTitle(column.column_name),
          items: tagPredicateItems(
            column.column_name,
            tagRows.map((row) => row.tag),
          ),
        };
      }
    }
    labelCharts = charts;
  }

  async function initialize(): Promise<void> {
    const wasm = await wasmConnector();
    dbConnector = wasm;
    coordinator.databaseConnector(wasm);
    await coordinator.exec(`CREATE OR REPLACE VIEW atlas_source AS SELECT * FROM read_parquet(${SQL.literal(atlasUrl)})`);
    // Small per-dataset lookup (task modality, domain, any further label), joined onto "dataset" —
    // not worth duplicating into the main parquet since it only varies per dataset, not per point.
    await loadDatasetClasses();
    const labelSelect = labelColumns.map((column) => `dataset_classes.${quoteIdent(column)}, `).join("");
    realMode = Boolean(selectionsUrl && runsUrl);
    if (!realMode) {
      await coordinator.exec(`
        CREATE OR REPLACE TABLE display_points AS
        SELECT atlas_source.*, ${labelSelect}${POINT_LABEL_SQL}, ${CHANNEL_GROUP_SQL}
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
        big_recording_index, candidate_k, selection_eta, actual_size, kernel_method,
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
      SELECT combined.*, ${labelSelect}${POINT_LABEL_SQL}, ${CHANNEL_GROUP_SQL} FROM (
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
    const membershipAggregates = DPP_FILTERS.flatMap((filter) =>
      (["top", "bottom"] as const).map((direction) => `
        list(DISTINCT selection_runs.w_interaction) FILTER (
          WHERE selection_runs.direction = '${direction}'
            AND selection_runs.selection_eta = ${filter.eta}
            AND selection_runs.kernel_method = '${filter.kernel}'
        ) AS ${dppColumn(direction, filter)}`),
    ).join(",");
    const membershipColumns = DPP_FILTERS.flatMap((filter) =>
      (["top", "bottom"] as const).map((direction) => {
        const column = dppColumn(direction, filter);
        return `coalesce(dpp_memberships.${column}, []::DOUBLE[]) AS ${column}`;
      }),
    ).join(",");
    await coordinator.exec(`
      CREATE OR REPLACE TABLE dpp_memberships AS
      SELECT selection_memberships.row_id, ${membershipAggregates}
      FROM selection_memberships
      INNER JOIN selection_runs USING (run_id)
      WHERE selection_runs.method = 'dpp'
      GROUP BY selection_memberships.row_id
    `);
    await coordinator.exec(`
      CREATE OR REPLACE TABLE display_points_with_dpp AS
      SELECT display_points.*, ${membershipColumns}
      FROM display_points
      LEFT JOIN dpp_memberships USING (row_id)
    `);
    activeTableName = "display_points_with_dpp";
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
  <main class:real-mode={realMode} class:curves-mode={realMode && railTab === "curves"} class="atlas-app">
    {#if realMode}
      <aside class="side-rail" aria-label="Atlas controls and active run summary">
        <div class="panel-section">
          <p class="eyebrow">REVE explorer</p>
          <h1>Embedding Atlas</h1>
          <p class="rail-copy">Switch among precomputed rankings and diversity selections.</p>
          <div class="rail-tabs" role="tablist" aria-label="Atlas side-panel view">
            <button type="button" role="tab" class:active={railTab === "explore"} aria-selected={railTab === "explore"} onclick={() => railTab = "explore"}>Explore</button>
            <button type="button" role="tab" class:active={railTab === "curves"} aria-selected={railTab === "curves"} disabled={curveMultiplicative.length <= 1 && curveAdditive.length <= 1} onclick={() => railTab = "curves"}>Operating curves</button>
          </div>
          {#if railTab === "explore"}
            <p class="display-note">Display cap: about 2M points; all selection members are retained.</p>
            <p class="explore-note">Filter the stable atlas by any precomputed DPP selection. These controls preserve the canvas and combine with the atlas charts.</p>
            {#if atlasMultiplicativePoints.length > 1 || atlasAdditivePoints.length > 1}
              <div class="mini-curve">
                <div class="mini-curve-header">
                  <h3>Vendi diversity / loss</h3>
                  <span class="mini-curve-w">{atlasWValue === null ? "" : `w=${atlasWValue}`}</span>
                </div>
                <div class="mini-curve-legend">
                  <button
                    type="button"
                    class="kernel-chip"
                    class:active={atlasDppKernel === "multiplicative"}
                    onclick={() => { if (atlasDppKernel !== "multiplicative") { atlasDppKernel = "multiplicative"; applyAtlasDppFilter(true); } }}
                    title="Select multiplicative kernel"
                  >
                    <span class="chip-swatch multiplicative" class:active={atlasDppKernel === "multiplicative"}></span>
                    <span>Multiplicative</span>
                  </button>
                  <button
                    type="button"
                    class="kernel-chip"
                    class:active={atlasDppKernel === "additive"}
                    onclick={() => { if (atlasDppKernel !== "additive") { atlasDppKernel = "additive"; applyAtlasDppFilter(true); } }}
                    title="Select additive kernel"
                  >
                    <span class="chip-swatch additive" class:active={atlasDppKernel === "additive"}></span>
                    <span>Additive</span>
                  </button>
                </div>
                <svg viewBox="0 0 420 270" role="img" aria-label="Vendi diversity versus mean reconstruction loss comparing multiplicative and additive DPP configurations">
                  <line class="axis-line" x1={CHART_LEFT} y1={CHART_TOP} x2={CHART_LEFT} y2={CHART_BOTTOM} />
                  <line class="axis-line" x1={CHART_LEFT} y1={CHART_BOTTOM} x2={CHART_RIGHT} y2={CHART_BOTTOM} />

                  <!-- Axis ticks and numerical values -->
                  <line class="axis-tick" x1={CHART_LEFT - 3} y1={CHART_TOP} x2={CHART_LEFT} y2={CHART_TOP} />
                  <text class="axis-val y-val" x={CHART_LEFT - 5} y={CHART_TOP + 4} text-anchor="end">{atlasLossMax.toFixed(2)}</text>
                  <line class="axis-tick" x1={CHART_LEFT - 3} y1={CHART_BOTTOM} x2={CHART_LEFT} y2={CHART_BOTTOM} />
                  <text class="axis-val y-val" x={CHART_LEFT - 5} y={CHART_BOTTOM} text-anchor="end">{atlasLossMin.toFixed(2)}</text>

                  <line class="axis-tick" x1={CHART_LEFT} y1={CHART_BOTTOM} x2={CHART_LEFT} y2={CHART_BOTTOM + 3} />
                  <text class="axis-val x-val" x={CHART_LEFT} y={CHART_BOTTOM + 14} text-anchor="start">{atlasVendiMin.toFixed(1)}</text>
                  <line class="axis-tick" x1={CHART_RIGHT} y1={CHART_BOTTOM} x2={CHART_RIGHT} y2={CHART_BOTTOM + 3} />
                  <text class="axis-val x-val" x={CHART_RIGHT} y={CHART_BOTTOM + 14} text-anchor="end">{atlasVendiMax.toFixed(1)}</text>

                  <!-- Shaded curve (unselected kernel) -->
                  {#if atlasShadedPoints.length > 1}
                    <polyline
                      class="mini-curve-line shaded {atlasDppKernel === 'multiplicative' ? 'additive' : 'multiplicative'}"
                      points={atlasShadedPoints.map((point) => `${point.x},${point.y}`).join(" ")}
                    />
                    {#each atlasShadedPoints as point}
                      <!-- svelte-ignore a11y_click_events_have_key_events -->
                      <!-- svelte-ignore a11y_no_static_element_interactions -->
                      <circle
                        class="mini-curve-point shaded {atlasDppKernel === 'multiplicative' ? 'additive' : 'multiplicative'}"
                        cx={point.x}
                        cy={point.y}
                        r="3.5"
                        onclick={() => selectAtlasPoint(point)}
                      ><title>{point.tooltip} (shaded — click to select)</title></circle>
                    {/each}
                  {/if}

                  <!-- Active curve (selected kernel) -->
                  {#if atlasActivePoints.length > 1}
                    <polyline
                      class="mini-curve-line active {atlasDppKernel}"
                      points={atlasActivePoints.map((point) => `${point.x},${point.y}`).join(" ")}
                    />
                    {#each atlasActivePoints as point}
                      <!-- svelte-ignore a11y_click_events_have_key_events -->
                      <!-- svelte-ignore a11y_no_static_element_interactions -->
                      <circle
                        class="mini-curve-point active {atlasDppKernel}"
                        class:current={point.current}
                        cx={point.x}
                        cy={point.y}
                        r={point.r}
                        onclick={() => selectAtlasPoint(point)}
                      ><title>{point.tooltip}</title></circle>
                    {/each}
                  {/if}
                </svg>
                <div class="axis-labels"><span>Vendi diversity →</span><span>loss ↑</span></div>
              </div>
            {/if}
            <div class="control-stack dpp-controls">
              <label>Points<select bind:value={atlasDppMode} onchange={() => applyAtlasDppFilter()}><option value="all">All points</option><option value="selected">Selected only</option></select></label>
              <div class="compact-controls">
                <label>Direction<select bind:value={atlasDppDirection} onchange={() => applyAtlasDppFilter(true)}><option value="bottom">Bottom loss</option><option value="top">Top loss</option></select></label>
                <label>Candidate pool<select bind:value={atlasDppEta} onchange={() => applyAtlasDppFilter(true)}><option value="eta_010">10%</option><option value="eta_001">1%</option></select></label>
              </div>
              <label>Kernel<select bind:value={atlasDppKernel} onchange={() => applyAtlasDppFilter(true)}><option value="multiplicative">Multiplicative</option><option value="additive">Additive</option></select></label>
              <label>Interaction weight
                <input type="range" min="0" max={Math.max(0, atlasWValues.length - 1)} step="1" bind:value={atlasWIndex} oninput={() => applyAtlasDppFilter()} disabled={atlasDppMode === "all" || atlasWValues.length <= 1} />
                <span class="sweep-value">{atlasWValue === null ? "—" : `w=${atlasWValue}`}</span>
              </label>
            </div>
          {/if}
        </div>
        {#if railTab === "curves"}
          <div class="panel-section">
            <p class="eyebrow">Aggregate analysis</p>
            <h2>Operating curves</h2>
            <p class="view-note">Curve controls are independent from the atlas selection filters.</p>
            <div class="curve-controls">
              <label>Direction<select bind:value={curveDirection}><option value="bottom">Bottom loss</option><option value="top">Top loss</option></select></label>
              <label>Candidate pool<select bind:value={curveEta}><option value="eta_010">10% per recording</option><option value="eta_001">1% per recording</option></select></label>
              <label>Kernel<select bind:value={curveKernel}><option value="multiplicative">Multiplicative</option><option value="additive">Additive</option></select></label>
            </div>
              {#snippet operatingCurveChart(
                title: string,
                points: ChartPoint[],
                shaded: ChartPoint[],
                baselinePoint: BaselinePoint | null,
                xAxisLabel: string,
                note: string,
                xMinVal: number,
                xMaxVal: number,
                xDecimals = 1,
              )}
              <div class="tradeoff">
                <h4>{title} / loss</h4>
                <svg viewBox="0 0 420 270" role="img" aria-label="{title} versus mean reconstruction loss comparing multiplicative and additive kernels versus stratified baseline">
                  <line class="axis-line" x1={CHART_LEFT} y1={CHART_TOP} x2={CHART_LEFT} y2={CHART_BOTTOM} />
                  <line class="axis-line" x1={CHART_LEFT} y1={CHART_BOTTOM} x2={CHART_RIGHT} y2={CHART_BOTTOM} />

                  <!-- Axis ticks and numerical values -->
                  <line class="axis-tick" x1={CHART_LEFT - 3} y1={CHART_TOP} x2={CHART_LEFT} y2={CHART_TOP} />
                  <text class="axis-val y-val" x={CHART_LEFT - 5} y={CHART_TOP + 4} text-anchor="end">{lossMax.toFixed(2)}</text>
                  <line class="axis-tick" x1={CHART_LEFT - 3} y1={CHART_BOTTOM} x2={CHART_LEFT} y2={CHART_BOTTOM} />
                  <text class="axis-val y-val" x={CHART_LEFT - 5} y={CHART_BOTTOM} text-anchor="end">{lossMin.toFixed(2)}</text>

                  <line class="axis-tick" x1={CHART_LEFT} y1={CHART_BOTTOM} x2={CHART_LEFT} y2={CHART_BOTTOM + 3} />
                  <text class="axis-val x-val" x={CHART_LEFT} y={CHART_BOTTOM + 14} text-anchor="start">{xMinVal.toFixed(xDecimals)}</text>
                  <line class="axis-tick" x1={CHART_RIGHT} y1={CHART_BOTTOM} x2={CHART_RIGHT} y2={CHART_BOTTOM + 3} />
                  <text class="axis-val x-val" x={CHART_RIGHT} y={CHART_BOTTOM + 14} text-anchor="end">{xMaxVal.toFixed(xDecimals)}</text>

                  {#if baselinePoint}
                    <line class="baseline-guide" x1={CHART_LEFT} y1={baselinePoint.y} x2={CHART_RIGHT} y2={baselinePoint.y} />
                    <line class="baseline-guide" x1={baselinePoint.x} y1={CHART_TOP} x2={baselinePoint.x} y2={CHART_BOTTOM} />
                  {/if}

                  <!-- Shaded curve (unselected kernel) -->
                  {#if shaded.length > 1}
                    <polyline
                      class="curve-line shaded {curveKernel === 'multiplicative' ? 'additive' : 'multiplicative'}"
                      points={shaded.map((point) => `${point.x},${point.y}`).join(" ")}
                    />
                    {#each shaded as point}
                      <circle
                        class="curve-point shaded {curveKernel === 'multiplicative' ? 'additive' : 'multiplicative'}"
                        cx={point.x}
                        cy={point.y}
                        r="3"
                      ><title>{point.tooltip} (shaded — switch kernel to inspect)</title></circle>
                    {/each}
                  {/if}

                  <!-- Active curve (selected kernel) -->
                  <polyline
                    class="curve-line active {curveKernel}"
                    points={points.map((point) => `${point.x},${point.y}`).join(" ")}
                  />
                  {#each points as point}
                    <circle
                      class="curve-point active {curveKernel}"
                      class:current={point.current}
                      cx={point.x}
                      cy={point.y}
                      r={point.r}
                    ><title>{point.tooltip}</title></circle>
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
                  <li>
                    <span class="swatch curve active {curveKernel}"></span>
                    <span>DPP sweep ({curveKernel})</span>
                  </li>
                  <li>
                    <span class="swatch curve shaded {curveKernel === 'multiplicative' ? 'additive' : 'multiplicative'}"></span>
                    <span class="shaded-label">DPP sweep ({curveKernel === 'multiplicative' ? 'additive' : 'multiplicative'})</span>
                  </li>
                  {#if baselinePoint}
                    <li><span class="swatch baseline"></span>Stratified baseline</li>
                  {/if}
                </ul>
                <small>{note}</small>
              </div>
              {/snippet}
              {#if curveMultiplicative.length > 1 || curveAdditive.length > 1}
                <div class="tradeoff-group">
                  <div class="tradeoff-header">
                    <h3>Operating curves</h3>
                    {#if strataBaseline}
                      <label class="baseline-toggle"><input type="checkbox" bind:checked={showBaseline} /> Show baseline</label>
                    {/if}
                  </div>
                  {@render operatingCurveChart(
                    "Vendi diversity",
                    vendiActivePoints,
                    vendiShadedPoints,
                    vendiBaselinePoint,
                    "Vendi diversity →",
                    curveKernel === "multiplicative" ? "Higher w emphasizes ranking utility." : "Higher w emphasizes interaction/diversity.",
                    vendiMin,
                    vendiMax,
                    1,
                  )}
                  {@render operatingCurveChart(
                    "Cosine similarity",
                    cosineActivePoints,
                    cosineShadedPoints,
                    cosineBaselinePoint,
                    "← more diverse · cosine similarity · less diverse →",
                    "Mean pairwise cosine of selected rows; lower means more diverse.",
                    cosineMin,
                    cosineMax,
                    2,
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
            include: [
              "recon_loss", "dataset", "n_channels", "modality",
              ...labelColumns,
              ...(realMode ? ["big_recording_index"] : []),
            ],
            override: {
              n_channels: {
                type: "predicates",
                title: "Channels",
                items: CHANNEL_PREDICATE_ITEMS,
              },
              modality: {
                type: "predicates",
                title: "Modality",
                items: [
                  { name: "EEG only", predicate: "modality = 'EEG'" },
                  { name: "MEG only", predicate: "modality = 'MEG'" },
                ],
              },
              ...labelCharts,
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
    --curve-additive: #0284c7;
    --curve-shaded: #94a3b8;
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
      --curve-additive: #38bdf8;
      --curve-shaded: #64748b;
      --shadow: rgb(0 0 0 / 45%);
      --status-card-bg: rgb(24 27 38 / 92%);
      --run-id-text: #838da3;
      --spinner-track: #333952;
    }
  }
  .atlas-app { display: grid; grid-template-columns: 20rem minmax(0, 1fr); min-width: 0; min-height: 0; width: 100%; height: 100%; overflow: hidden; background: var(--bg); }
  .atlas-app.curves-mode { grid-template-columns: minmax(28rem, 34rem) minmax(0, 1fr); }
  .atlas-app:not(.real-mode) { display: block; }
  .atlas-shell { min-width: 0; min-height: 0; width: 100%; height: 100%; overflow: hidden; position: relative; }
  .side-rail { z-index: 2; box-sizing: border-box; display: flex; flex-direction: column; min-width: 0; min-height: 0; overflow: auto; border-right: 1px solid var(--border); background: var(--panel-bg); }
  .panel-section { box-sizing: border-box; padding: 1.25rem 1rem; }
  .panel-section + .panel-section { border-top: 1px solid var(--border); }
  h1 { margin: 0 0 0.55rem; color: var(--heading); font-size: 1.45rem; letter-spacing: -0.04em; }
  h2 { margin: 0.1rem 0 0.3rem; color: var(--heading); font-size: 1.15rem; }
  h3 { margin: 1.25rem 0 0.25rem; color: var(--text-muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }
  p { margin: 0; color: var(--text-muted); line-height: 1.45; }
  .rail-copy { margin-bottom: 0.75rem; font-size: 0.78rem; }
  .rail-tabs { display: grid; grid-template-columns: 1fr 1fr; gap: 0.25rem; margin-bottom: 1rem; padding: 0.2rem; border-radius: 0.55rem; background: var(--input-disabled-bg); }
  .rail-tabs button { margin: 0; padding: 0.5rem 0.65rem; border: 0; border-radius: 0.4rem; color: var(--text-soft); background: transparent; font-size: 0.72rem; font-weight: 650; cursor: pointer; }
  .rail-tabs button.active { color: var(--hero-text); background: var(--panel-bg); box-shadow: 0 1px 3px var(--shadow); }
  .rail-tabs button:disabled { color: var(--input-disabled-text); cursor: not-allowed; opacity: 0.65; }
  .display-note { margin-bottom: 1.25rem; color: var(--text-faint); font-size: 0.68rem; }
  .explore-note { margin-bottom: 1rem; font-size: 0.78rem; }
  .eyebrow { margin-bottom: 0.45rem; color: var(--eyebrow); font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; }
  .control-stack { display: grid; gap: 0.75rem; }
  .compact-controls { display: grid; grid-template-columns: 1fr 1fr; gap: 0.65rem; }
  .dpp-controls { padding-top: 1rem; border-top: 1px solid var(--border); }
  .curve-controls { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.65rem; margin-bottom: 1rem; }
  label { display: grid; gap: 0.25rem; color: var(--text-muted); font-size: 0.68rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }
  select { width: 100%; min-width: 0; box-sizing: border-box; padding: 0.45rem 0.4rem; border: 1px solid var(--input-border); border-radius: 0.35rem; color: var(--input-text); background: var(--input-bg); font-size: 0.75rem; text-transform: none; }
  select:disabled { color: var(--input-disabled-text); background: var(--input-disabled-bg); }
  input[type="range"] { width: 100%; accent-color: var(--accent); }
  input[type="range"]:disabled { opacity: 0.5; }
  .sweep-value { color: var(--text-soft); font-size: 0.68rem; font-weight: 500; letter-spacing: normal; text-transform: none; }
  .view-note { margin-bottom: 1rem; color: var(--text-soft); font-size: 0.68rem; }
  small, .axis-labels { color: var(--text-soft); font-size: 0.65rem; }
  .mini-curve { margin-bottom: 1rem; padding: 0.65rem; border: 1px solid var(--border); border-radius: 0.55rem; background: var(--input-disabled-bg); }
  .mini-curve-header { display: flex; align-items: baseline; justify-content: space-between; gap: 0.5rem; }
  .mini-curve-header h3 { margin: 0; }
  .mini-curve-header .mini-curve-w { color: var(--hero-text); font-size: 0.7rem; font-weight: 650; }
  .mini-curve-legend { display: flex; gap: 0.35rem; margin: 0.4rem 0 0.35rem; }
  .kernel-chip { display: inline-flex; align-items: center; gap: 0.35rem; margin: 0; padding: 0.18rem 0.45rem; border: 1px solid var(--input-border); border-radius: 0.35rem; background: var(--input-bg); color: var(--text-faint); opacity: 0.55; font-size: 0.65rem; font-weight: 500; cursor: pointer; transition: all 0.15s ease; }
  .kernel-chip:hover { border-color: var(--text-soft); color: var(--heading); opacity: 0.85; }
  .kernel-chip.active { border-color: var(--accent); background: var(--hero-bg); color: var(--hero-text); opacity: 1; font-weight: 650; }
  .chip-swatch { display: inline-block; width: 0.45rem; height: 0.45rem; border-radius: 0.12rem; background: var(--curve-shaded); opacity: 0.3; transition: all 0.15s ease; }
  .chip-swatch.active.multiplicative { background: var(--accent); opacity: 1; }
  .chip-swatch.active.additive { background: var(--curve-additive); opacity: 1; }

  .mini-curve svg { display: block; width: 100%; height: auto; aspect-ratio: 420 / 270; overflow: visible; }
  .mini-curve line.axis-line { stroke: var(--input-border); stroke-width: 1; }
  .mini-curve line.axis-tick { stroke: var(--input-border); stroke-width: 1; }
  .mini-curve text.axis-val { fill: var(--text-faint); font-size: 0.62rem; font-family: inherit; font-variant-numeric: tabular-nums; user-select: none; }

  .mini-curve polyline.mini-curve-line { fill: none; }
  .mini-curve polyline.mini-curve-line.active.multiplicative { stroke: var(--accent); stroke-width: 2; }
  .mini-curve polyline.mini-curve-line.active.additive { stroke: var(--curve-additive); stroke-width: 2; }
  .mini-curve polyline.mini-curve-line.shaded { stroke: var(--curve-shaded); stroke-width: 1.25; stroke-dasharray: 3 3; opacity: 0.2; transition: opacity 0.15s ease; }
  .mini-curve:hover polyline.mini-curve-line.shaded { opacity: 0.32; }

  .mini-curve circle.mini-curve-point.active.multiplicative { fill: var(--accent); cursor: pointer; }
  .mini-curve circle.mini-curve-point.active.additive { fill: var(--curve-additive); cursor: pointer; }
  .mini-curve circle.mini-curve-point.active.current { fill: var(--eyebrow); stroke: var(--panel-bg); stroke-width: 2; cursor: pointer; }
  .mini-curve circle.mini-curve-point.shaded { fill: var(--curve-shaded); opacity: 0.2; cursor: pointer; transition: all 0.15s ease; }
  .mini-curve circle.mini-curve-point.shaded:hover { fill: var(--text-soft); opacity: 0.75; }

  .tradeoff-group { margin-top: 1.25rem; }
  .tradeoff-header { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 0.5rem; }
  .tradeoff-header h3 { margin: 0; }
  .baseline-toggle { display: flex; align-items: center; gap: 0.3rem; color: var(--text-soft); font-size: 0.65rem; font-weight: 400; letter-spacing: normal; text-transform: none; }
  .baseline-toggle input { accent-color: var(--curve-baseline); }
  .baseline-note { display: block; margin-top: 0.35rem; }
  .tradeoff { margin-top: 0.75rem; }
  .tradeoff h4 { margin: 0 0 0.3rem; color: var(--text-muted); font-size: 0.68rem; font-weight: 600; text-transform: none; }
  .tradeoff svg { display: block; width: 100%; height: auto; aspect-ratio: 420 / 270; overflow: visible; }
  .tradeoff line.axis-line { stroke: var(--input-border); stroke-width: 1; }
  .tradeoff line.axis-tick { stroke: var(--input-border); stroke-width: 1; }
  .tradeoff text.axis-val { fill: var(--text-faint); font-size: 0.62rem; font-family: inherit; font-variant-numeric: tabular-nums; user-select: none; }
  .tradeoff line.baseline-guide { stroke: var(--curve-baseline); stroke-width: 1; stroke-dasharray: 2 2; opacity: 0.55; }
  .tradeoff line.baseline-whisker { stroke: var(--curve-baseline); stroke-width: 1.25; }
  .tradeoff polyline.curve-line { fill: none; }
  .tradeoff polyline.curve-line.active.multiplicative { stroke: var(--accent); stroke-width: 1.75; }
  .tradeoff polyline.curve-line.active.additive { stroke: var(--curve-additive); stroke-width: 1.75; }
  .tradeoff polyline.curve-line.shaded { stroke: var(--curve-shaded); stroke-width: 1.25; stroke-dasharray: 3 3; opacity: 0.2; transition: opacity 0.15s ease; }
  .tradeoff:hover polyline.curve-line.shaded { opacity: 0.32; }
  .tradeoff circle.curve-point.active.multiplicative { fill: var(--accent); }
  .tradeoff circle.curve-point.active.additive { fill: var(--curve-additive); }
  .tradeoff circle.curve-point.active.current { fill: var(--eyebrow); stroke: var(--panel-bg); stroke-width: 1.5; }
  .tradeoff circle.curve-point.shaded { fill: var(--curve-shaded); opacity: 0.2; cursor: pointer; transition: all 0.15s ease; }
  .tradeoff circle.curve-point.shaded:hover { fill: var(--text-soft); opacity: 0.75; }
  .tradeoff rect.baseline-marker { fill: var(--curve-baseline); stroke: var(--panel-bg); stroke-width: 1; }
  .tradeoff .legend { display: flex; flex-wrap: wrap; gap: 0.6rem; margin: 0.5rem 0 0.35rem; padding: 0; list-style: none; color: var(--text-muted); font-size: 0.65rem; }
  .tradeoff .legend li { display: flex; align-items: center; gap: 0.3rem; }
  .tradeoff .swatch { display: inline-block; width: 0.55rem; height: 0.55rem; border-radius: 0.15rem; }
  .tradeoff .swatch.curve.multiplicative { background: var(--accent); }
  .tradeoff .swatch.curve.additive { background: var(--curve-additive); }
  .tradeoff .swatch.shaded { background: var(--curve-shaded); opacity: 0.25; }
  .tradeoff .shaded-label { color: var(--text-faint); opacity: 0.65; }
  .tradeoff .swatch.baseline { background: var(--curve-baseline); transform: rotate(45deg); }
  .axis-labels { display: flex; justify-content: space-between; }
  .status { box-sizing: border-box; display: grid; width: 100%; height: 100%; place-items: center; padding: 2rem; background: radial-gradient(circle at 20% 20%, rgb(86 101 255 / 14%), transparent 32rem), radial-gradient(circle at 80% 70%, rgb(34 197 175 / 12%), transparent 28rem), var(--bg); }
  .status-card { width: min(42rem, 100%); padding: 2.5rem; border: 1px solid var(--border); border-radius: 1rem; background: var(--status-card-bg); box-shadow: 0 1.25rem 4rem var(--shadow); }
  .spinner { width: 1.75rem; height: 1.75rem; margin-bottom: 1.25rem; border: 3px solid var(--spinner-track); border-top-color: var(--accent); border-radius: 999px; animation: spin 0.8s linear infinite; }
  button { margin-top: 1.5rem; padding: 0.65rem 1rem; border: 0; border-radius: 0.5rem; color: white; background: var(--accent-strong); cursor: pointer; }
  @media (max-width: 1050px) { .atlas-app.real-mode { grid-template-columns: 16rem minmax(0, 1fr); } .atlas-app.real-mode.curves-mode { grid-template-columns: minmax(24rem, 28rem) minmax(0, 1fr); } }
  @media (max-width: 700px) { .atlas-app.real-mode { display: block; overflow: auto; } .side-rail { border: 0; border-bottom: 1px solid var(--border); } .atlas-shell { height: 70vh; min-height: 32rem; } }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
