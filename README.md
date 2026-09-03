# Path-centered multiscale emergence attribution

This repository implements the frozen confirmatory protocol for Schelling and
Deffuant simulations. The scientific object is a complete, prospectively
specified `parameter -> Micro -> Meso -> Macro` mechanism path. Candidate
relations are deterministic adjacent-scale projections of those paths; they
are not independently proposed discoveries.

The evidence boundary is explicit:

1. An LLM constructs executable Micro, Meso, and Macro observables using only
   public simulator semantics and the public raw-log schema.
2. The selected 16/8/4 observable set is frozen. A separate LLM call may only
   combine its exact IDs into 16--24 complete testable paths per scenario.
3. Natural trajectories temporally qualify a frozen path only when both of its
   adjacent relations pass the unchanged Stage 2 procedure.
4. Matched-seed simulator-parameter interventions classify the same frozen
   path as `supported`, `contradicted`, `inconclusive`, or
   `manipulation_failure`.

No LLM call sees baseline values, temporal results, intervention results,
Controlled Recovery truth, hidden channels, or earlier experiment outcomes.
The program never attempts fuzzy semantic matching between different
observables.

## Frozen protocol

- Indicator capacity: 16 Micro, 8 Meso, and 4 Macro per scenario.
- Indicator calls: 3 selection-eligible plus 3 replication-only calls.
- Path calls: 1 primary plus 2 replication-only calls over the same frozen IDs.
- Path capacity: 16--24; at least 4 paths per controllable parameter and at
  least 2 paths per Macro endpoint; duplicate triples are rejected.
- Prospective predictions: 6 per scenario, each bound to a frozen
  `candidate_path_id` before baseline simulation.
- FDR families: pre-data Macro-outcome hypothesis groups. A shared fitted
  relation can receive a separate q-value in every group in which it occurs.
- Primary seeds: 3101--3124. Independent holdout seeds: 4101--4112.
- Formal workload: 864 simulator trajectories. The structural upgrade expands
  hypotheses, not simulator runs.

The simulator core, Stage 2 lagged OLS, lag 1--5, parent alpha 0.10, FDR alpha
0.05, 100 trajectory bootstraps, support threshold 0.65, Stage 3 paired
bootstrap 500, 95% CI, standardized-effect threshold 0.10, onset start 0,
four-step onset rule, evaluation start 15, terminal window 24, lag tolerance 2,
matched seeds, dose response, holdout, negative control, observation
robustness, data efficiency, and Controlled Recovery machinery are unchanged.

## Installation and formal command

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test]"
Copy-Item config\llm_api.example.json config\llm_api.local.json
```

Fill the local API configuration, then use a new run ID:

```powershell
.\.venv\Scripts\python run_experiment.py `
  --config config\experiment.json `
  --llm-config config\llm_api.local.json `
  --run-id confirmatory_path_v4_001 `
  --workers auto `
  --plot-repo "..\llm-agentic-dis"
```

Add `--resume` only for the same immutable run. A checkpoint or frozen-artifact
hash mismatch fails closed. Once baseline simulation exists, semantic stages
cannot be rerun.

The formal order is:

```text
indicator_generation -> indicator_freeze -> path_generation -> semantic_freeze
-> baseline_simulation -> temporal -> path_temporal_qualification
-> intervention_simulation -> intervention -> path_intervention_classification
-> prospective -> primary_freeze -> dose_response -> holdout_simulation
-> holdout_confirmation -> temporal_negative_control -> robustness -> export
-> render
```

`prospective` in this list evaluates predictions already generated and hashed
inside Phase B; it does not create or select predictions after seeing results.

## Main artifacts

- `representation/indicators_frozen.json` and `INDICATORS_FROZEN.sha256`
- `representation/candidate_paths.json` and
  `CANDIDATE_PATHS_FROZEN.sha256`
- `representation/derived_candidate_edges.json`
- `representation/prospective_predictions.json` and
  `PROSPECTIVE_PREDICTIONS_FROZEN.sha256`
- indicator and path replication CSV/JSON files under `representation/`
- `analysis/path_temporal_qualification.csv`
- `analysis/path_intervention_classification.csv`
- `analysis/path_funnel_summary.csv`
- `analysis/path_dose_response_summary.csv`
- `analysis/path_temporal_negative_control.csv`
- `analysis/holdout_path_confirmation.csv`
- `analysis/prospective_validation.csv`
- `analysis/representative_path_selection.json`
- `analysis/attribution_objects.json`

Figure 5 and Figure 7 consume only genuine supported frozen paths. If none
exist, the renderers report that outcome and do not reconstruct a path from
unrelated evidence.

## Verification without scientific execution

```powershell
python -m pytest -q
python smoke_pipeline.py --run-id smoke_local --workers 1
python run_dev_e2e.py --run-id dev_local --workers 2
```

Smoke and dev outputs are marked `NON_SCIENTIFIC`; they verify contracts and
wiring only. A dev run with zero supported paths is still an engineering pass
when all frozen-data contracts hold. No mock result is formal evidence.

The previously inspected `res_f` outputs and seed pools 1101--1124 /
2101--2112 remain development evidence and must not be reused or merged into
the new confirmatory run.
