# Implementation plan

## Scientific contract

The new project implements one immutable three-stage chain:

1. **Semantic hypothesis.** Before any formal simulation statistic is inspected, an OpenAI-compatible language model receives only the scenario description, agent rules, controllable parameters, raw-log field schema, a generic safe computation grammar, and a fixed 16/8/4 representation budget. It produces executable indicators, scale and branch assignments, candidate adjacent-scale relations, parameter associations, and prospective falsifiable predictions.
2. **Temporal evidence.** Repeated baseline trajectories qualify only the proposed micro-to-meso and meso-to-macro relations with standardised level-based lagged OLS. Every target model contains lags 1--5 of the target itself and all competing candidate parents. A joint screen is followed by conditional refitting, branch-local Benjamini-Hochberg correction, and whole-trajectory bootstrap support.
3. **Intervention evidence.** Simulator parameters, never indicator values, are changed to fixed minus/baseline/plus levels under matched seeds. The analysis reports raw paired effects and separately standardised effects, paired-seed bootstrap intervals, onset, propagation delay, observational lag, direction agreement, non-local manipulations, and all positive or negative evidence states.

The final attribution keeps semantic hypotheses, observational temporal qualification, and simulator intervention evidence separate. A temporally qualified graph is not described as a complete causal graph.

## First-stage redesign

The previous implementation exposed a complete executable indicator catalogue before the model call and validated model output against those predefined indicator identities. It also contained deterministic completion of missing parameter associations. Neither behaviour is retained.

The new first stage instead:

- exposes raw field schemas and generic operators, not named indicators;
- asks the model to generate the complete indicator content and safe JSON AST computations;
- validates JSON structure, real fields, types, AST operators, exact scale budgets, branch completeness, adjacent-scale edges, parameter names, and duplicate identifiers;
- permits repair only with validation errors and the rejected object, never with baseline statistics, reference information, temporal results, intervention results, or evaluation metrics;
- runs three independent generations per scenario and applies a frozen data-blind selection key based on validity, source-field diversity, branch completeness, repair count, and canonical JSON hash;
- never inserts an association that the model did not propose;
- freezes the selected representation and prospective predictions before temporal or intervention analysis.

## Reusable ideas and rejected legacy behaviour

Reusable method-level ideas are counter-based random streams for matched simulator conditions, task-level process parallelism, whole-trajectory resampling, target self-history plus competing-parent regression, paired-seed uncertainty, stage checkpoints, and the plotting repository's typography/colour configuration.

The project does not copy the old package, static indicator catalogue, frozen representations, raw data, derived tables, graphs, intervention outputs, provider-specific clients, automatic association completion, whole-workflow config gate for raw simulation reuse, hard-coded publication numbers, frozen row counts, historical hashes, or plotting choices that hide false edges, mislabel a mean as a median, or silently clip colour values.

## Project structure

```text
_code/
  config/
    experiment.json
    llm_api.example.json
    predefined_observable_baseline.json
  src/emergence_attribution/
    schemas.py                 # structured representation and prediction models
    dsl.py                     # safe AST validation, typing, normalisation, execution
    raw_schemas.py             # prompt-safe field schemas and scenario/rule descriptions
    llm_client.py              # the only network/API access module
    semantic.py                # prompt, repairs, selection, freeze and agreement
    simulators.py              # Schelling, Deffuant and deterministic toy simulator
    simulation.py              # 384-task matrix, persistence and resume
    temporal.py                # two-stage OLS, BH-FDR and trajectory bootstrap
    interventions.py           # raw/standardised paired effects and classifications
    prospective.py             # post-experiment prediction validation
    evaluation.py              # evaluation-only alignment and metrics
    reference_truth.py         # simulator/evaluation-only controlled reference structures
    robustness.py              # ablations, sensitivity and scalability
    provenance.py              # immutable run contracts and SHA256 manifests
    progress.py                # Rich display and progress JSONL
    exporting.py               # paper-data and visualisation bundle schemas
    rendering.py               # dynamic Figure 2--8 renderer using external style config
    pipeline.py                # stage orchestration and checkpoints
  tests/
  run_experiment.py
  export_visualization_bundle.py
  render_paper_figures.py
  RUNBOOK.md
  IMPLEMENTATION_REPORT.md
```

## Data flow and isolation

```text
model rules + parameters + raw schema + safe grammar
                    |
                    v
     three independent model generations
                    |
          schema-only validation/repair
                    |
      frozen representation + predictions
                    |
 raw simulator logs +---------+------------------+
        |                      |                  |
        v                      v                  v
 baseline indicator       complete indicator   evaluation-only
 trajectories             trajectories         signature alignment
        |                      |                  |
        v                      v                  v
 lagged OLS + BH +        paired effects +     Edge F1/SHD/
 trajectory bootstrap     onset/classification lag metrics
        +----------------------+------------------+
                               |
                               v
             evidence integration and paper-data exports
```

The reference module is imported only by simulator internals and final evaluation. Prompt construction, representation validation and selection, temporal fitting, bootstrap filtering, intervention classification, and prospective prediction never import or read reference records.

Alignment uses a canonical computation signature plus a sorted source-field signature and result kind. A generated indicator maps only when exactly one controlled reference process has the same signature. Zero or multiple matches are recorded as `unmatched`; branch names, downstream outcomes, F1, lag, sign, and intervention results are never alignment inputs.

## Parallel and resume strategy

- Simulator seed-condition tasks, trajectory-bootstrap replicates, paired-bootstrap replicates, robustness repetitions, and scalability jobs are independent process-pool jobs.
- Every job seed is derived from a frozen master seed plus a stable task label; no mutable global random generator is shared.
- Results are sorted by task or replicate identifier before aggregation, making `workers=1` and multi-worker execution identical.
- API generations use bounded concurrent I/O but are persisted round by round.
- Raw files and every stage output have a hash-bearing checkpoint. `--resume` verifies the run contract and completed artifact hashes before skipping work.
- Changing source, prompt, experiment settings, or model settings invalidates resume and requires a new run identifier.

## Plotting-repository interface

`export_visualization_bundle.py` writes the established core filenames and required fields from the new run, but it derives row counts, indicator identifiers, branch labels, and hashes dynamically. It also emits `figure_inputs.generated.json`, `SHA256SUMS`, and `render_manifest.json`.

`render_paper_figures.py` reads visual style settings from the supplied local plotting repository and uses a generic data-root renderer so no frozen-run path or static node catalogue is required. False/correct/missed edges use clearly visible opacity, intervention curves and captions consistently use the mean estimator, and effect matrices use the complete value range with no undisclosed percentile clipping.

## Verification boundary

The implementation will include unit tests, a deterministic toy simulator, and a smoke wiring path. These outputs are labelled non-scientific and are physically separated from formal runs. Per the current instruction, no formal simulation, model call, temporal analysis, intervention analysis, robustness experiment, or paper rendering will be executed during implementation; only non-experimental source checks and unit-level software validation may be run.
