# Implementation report

## Completion state

The new experiment project has been implemented from an empty target directory. No historical raw trajectory, model response, representation, graph, intervention output, table value, figure value, row count, or hash is reused.

Current verification state:

- source compilation: **passed**;
- unit tests: **17 passed**;
- single-worker versus two-worker bootstrap determinism: **passed**;
- case-insensitive prohibited-name scan: **0 matches**;
- formal simulator/API/temporal/intervention/robustness run: **not executed by instruction**;
- deterministic toy smoke runner: **implemented but not executed by instruction**.

The test environment used an existing Python environment with the declared dependencies. An attempted minimal `pytest` install in another Python environment failed because that environment could not establish a valid TLS connection to the package index; no scientific task was affected.

## Files created

All files are new. The implementation contains:

- root commands: `run_experiment.py`, `export_visualization_bundle.py`, `render_paper_figures.py`, `smoke_pipeline.py`;
- configuration: `config\experiment.json`, `config\llm_api.example.json`, `config\semantic_prompt.txt`, `config\predefined_observable_baseline.json`;
- package and dependencies: `pyproject.toml`, `requirements.txt`, `.gitignore`;
- implementation package: `src\emergence_attribution\` with schema, DSL, prompt/API, simulators, simulation persistence, temporal analysis, intervention analysis, prospective validation, evaluation-only alignment, robustness, provenance, progress, export, rendering and orchestration modules;
- tests: 17 unit tests under `tests\` covering all requested contract categories;
- documentation: `README.md`, `IMPLEMENTATION_PLAN.md`, `RUNBOOK.md`, this report.

## New first stage

The model receives only the scenario description, behaviour rules, controllable parameter descriptions and fixed levels, raw field schema, generic safe grammar, representation budget and structural constraints. It receives no baseline numerical summary, reference structure, fitted graph, lag, F1/SHD, bootstrap result or intervention response.

For each scenario, three independent calls are made. Every generation saves:

- system/user prompt;
- every request and raw response;
- parsed JSON;
- schema/DSL/structure validation errors;
- repair-round metadata and token usage;
- accepted or rejected generation result.

Repair feedback contains only the rejected object and schema/executability errors. Selection is frozen before analysis and ranks only validity, source-field diversity, complete-path count, repair count and canonical JSON hash. It cannot read simulation statistics or evaluation results.

Parameter associations are part of the model output. The validator checks parameter and indicator existence and requires configured parameter coverage before freeze; it never inserts a missing association.

## Structured indicator DSL

Every computation is a JSON AST. The white list covers numeric arithmetic, safe ratio, comparison/filtering, reductions, quantiles, dispersion, entropy, temporal difference/rolling mean, spatial component statistics, spatial neighbour agreement, network assortativity, network density and channel selection.

Validation is recursive and checks:

- operator membership;
- raw-field existence;
- dtype compatibility;
- named-axis compatibility;
- reduction and selection parameters;
- exact source-field declaration;
- final numeric `[time]` result.

Execution uses NumPy and audited graph/spatial routines. There is no `eval`, `exec`, generated Python, provider-specific computation path or arbitrary import.

## Prospective prediction freeze

Each semantic generation includes prospective predictions. The selected predictions are written once to `representation\prospective_predictions.json`, and their SHA256 is recorded in `representation_validation.json` before any downstream stage.

Post-experiment validation first rechecks the frozen hash, then writes `analysis\prospective_validation.csv` with exactly these states:

- `supported`;
- `partially_supported`;
- `contradicted`;
- `manipulation_failure`;
- `inconclusive`.

Prediction outcomes never change thresholds, candidate relations, lags, intervention levels or generation selection.

## Stage 2 implementation

The temporal module implements the paper's level-based model rather than correlation, peak matching, first differences or another discovery algorithm.

For each target it precomputes per-trajectory design blocks containing:

- target lags 1--5;
- all proposed competing-parent lags 1--5;
- preserved trajectory boundaries.

The first joint OLS screen uses `parent_alpha=0.10`. Surviving candidate terms are refitted with all target self-history terms and all competing survivors. Coefficients, standard errors, Student-t statistics and two-sided p-values are calculated from standardised OLS. Benjamini-Hochberg correction is applied within each generated branch, and the minimum-q significant lag is retained per source-target pair.

The point graph must satisfy `q < 0.05`. One hundred whole-trajectory bootstrap replicates independently rerun the full qualification. The final graph additionally requires support at least 0.65 and saves edge sets, support, lag support and lag standard deviation.

## Stage 3 implementation

The simulator task matrix is two scenarios times eight conditions times 24 seeds, for 384 new raw tasks. Conditions are baseline, three parameter-minus, three parameter-plus and one mechanism-disabled variant. No indicator is directly manipulated.

Matched conditions use counter-derived streams based on scenario seed, step and stream identifier. Effects retain two separate estimands:

- raw paired response: intervention minus baseline, averaged over matched seeds;
- standardised paired response: raw paired difference divided by the sample SD over all baseline seed-time values from `evaluation_start=15` onward.

The baseline SD definition is frozen in config and exported with every summary. Five hundred paired-seed bootstrap replicates provide 95% intervals. Onset is frozen at step 0 with a minimum standardised effect of 0.10 and four consecutive significant steps. Cumulative evaluation begins at step 15.

Relation classification reports source manipulation, downstream stability, direction, onset order, propagation delay, observational lag and lag difference. Evidence states are `supported`, `directionally_contradicted`, `no_stable_downstream_effect`, `manipulation_failure` and `inconclusive`. Multi-source parameter associations are retained with `intervention_scope=non_local`.

## Controlled reference isolation and alignment

The controlled structures live in one evaluation-only module. Only simulator internals and final evaluation import it. Semantic generation, validation, selection, temporal fitting, bootstrap filtering, intervention classification and prospective validation do not.

Generated indicators align only through canonical computation, sorted source-field signature and temporal aggregation. Branch identifiers, fitted edges, lags, signs, F1, SHD and intervention outcomes are prohibited alignment inputs. Zero or multiple candidates are recorded as unmatched or ambiguous. The benchmark currently defines 12 controlled process identities rather than secretly treating all 28 generated indicators as a fixed historical catalogue.

## Ablations and robustness

The implementation retains:

- semantic proposal;
- unrestricted temporal search;
- single trajectory;
- trajectory voting;
- full method;
- without structured representation, reported as N/A rather than mixed with another method;
- without joint trajectories;
- without bootstrap;
- without paired seeds;
- model-generated executable observables versus a fixed-observable ablation comparator;
- trajectory-count sensitivity;
- observation noise;
- missing values;
- support-threshold sensitivity;
- five representation-corruption operators;
- mechanism-disabled checks;
- candidate-space scalability.

The missing-value robustness condition masks the pre-specified fraction first, then applies deterministic within-trajectory linear interpolation with boundary extension before the unchanged OLS procedure. This prevents listwise deletion across a large lagged parent design from silently eliminating nearly every row; the masking fraction and factor label remain in the exported record.

The fixed-observable comparator is built in a separate module, labelled ablation-only in config, and never enters the full-method semantic stage.

## Parallel and performance design

CPU work uses process pools for simulator tasks, trajectory bootstrap, paired bootstrap and robustness. API generations use a bounded thread pool. Every replicate has a stable seed derived from the master seed and explicit task label; aggregation is sorted by task or replicate identifier. No mutable global random generator is shared.

Performance measures include precomputed per-trajectory lagged blocks, vectorised paired-effect matrices, one bootstrap index matrix per intervention condition, cached standardisation within each fit, and process-worker initialisation that transfers temporal blocks once per worker rather than once per replicate.

Expected bottlenecks remain:

1. 384 full simulator tasks and compressed NPZ I/O;
2. six real model generations plus any repair latency;
3. whole-trajectory bootstrap repeated across data-efficiency and robustness conditions;
4. unrestricted and candidate-scaling OLS design matrices;
5. 500 paired-seed bootstrap curves across every represented indicator.

No repetition count, lag range, significance method, robustness condition or scientific estimand is reduced for speed.

## Checkpoint, progress and provenance

Each raw task uses atomic NPZ replacement plus a SHA256 sidecar. Resume verifies existing files and skips valid tasks. Each stage has a hash-bearing checkpoint. Source or contract drift forces a new run-id.

The Rich terminal display and `logs\progress.jsonl` record overall/stage progress, counts, percentage, elapsed, ETA, workers and current detail. `analysis\timing_summary.json` records stage durations.

The final run contains frozen experiment and redacted model config, source manifest, environment, stage manifests, artifact hashes and `RUN_FROZEN`. Existing completed run identifiers cannot be overwritten.

## Visualisation interface and corrections

`create_visualization_bundle` exports the established core filenames and required fields with dynamic row counts and actual indicator names. It generates `figure_inputs.generated.json`, `SHA256SUMS` and `render_manifest.json` for the new run.

The renderer reads the local plotting repository's export settings but uses a generic data-root adapter. This avoids frozen paths and static historical node semantics. It generates PNG/SVG/PDF and optional TIFF for Figure 2--8.

The known plotting issues are addressed explicitly:

- added relations use high-opacity red edges and missed counts are disclosed;
- intervention curves and titles consistently use the mean estimator;
- effect matrices use the full symmetric observed range, with no undisclosed percentile clipping.

## Remaining user decisions

Only operational choices remain:

1. fill the API base URL, key and model;
2. choose a fresh run-id and worker count;
3. supply the local plotting-repository path and desired output formats;
4. decide how unmatched generated indicators should be discussed in the manuscript after the run. They must not be manually remapped inside the frozen run; any changed alignment policy requires a new run-id.

No result-dependent threshold, onset, lag, selection rule or intervention value remains open.
