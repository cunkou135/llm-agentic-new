# Final pre-run audit

Audit date: 2026-09-02

Scope: current project only. No real LLM API, formal 24-seed experiment,
formal 100-repetition trajectory bootstrap suite, or formal 500-repetition
paired bootstrap suite was run. All cited executions are NON_SCIENTIFIC.

## Final correction gates

| Gate | Issue | Old behavior | New behavior | Test | Status |
|---|---|---|---|---|---|
| Mechanism-disabled semantic alignment | Schelling public and hidden paths disabled different mechanisms. | Public simulation removed destination preference while hidden recovery disabled relocation. | `disable_homophilic_relocation` targets the `destination_similarity -> s_meso_3 -> s_macro_3` homophilic-destination-selection branch. Deffuant `disable_backfire` targets the repulsion branch in both paths. | `test_schelling_public_and_hidden_disabled_mechanism_are_aligned`; `test_deffuant_public_and_hidden_disabled_mechanism_are_aligned`; dev gate | PASS |
| Representation robustness denominator | Mutated candidate counts could be overwritten by the original count. | Qualification used the original structured denominator. | `_metric_row` accepts the actual candidate count; collision-safe metric merging prevents silent duplicate-key replacement. | Candidate-count, qualification-denominator, and duplicate-key regression tests | PASS |
| Single-trajectory stability semantics | Resampling one trajectory could look perfectly stable. | n=1 bootstrap support could be reported as stability 1. | n=1 retains the point graph and qualification rate; stability, lag support/SD, and stability intervals are missing with `stability_estimable=false`. | n=1 semantics and Figure 4 tests; dev CSV gate | PASS |
| Undirected assortativity | Sorted endpoint IDs affected the numeric correlation. | Only stored `(min_id,max_id)` orientation was used. | Both orientations of every undirected edge are included; constant input has the fixed result 0. | Relabel-invariance and constant-value tests; dev gate | PASS |
| Prospective exact required path | Required edges could contain missing or extra relations. | The expected path only had to be included. | Required edges must exactly equal adjacent frozen temporal-order edges and all must be candidate edges. | Exact, missing, extra, and non-candidate cases | PASS |
| Random perturbation repetitions | Cross-branch errors could reuse the same prefix in every repetition. | `eligible[:count]` made repeated error sets identical. | Seeded sampling without replacement is deterministic within a repetition and distinct across repetitions; all other perturbation operators were also checked. | Cross-branch and all-operator repetition tests; dev candidate-set hashes | PASS |
| Edge-level intervention aggregation | One support could hide an explicit contradiction. | Controlled F1 accepted any supported attempt. | `not_applicable` is excluded; contradiction precedes support; otherwise one support is sufficient; all failures remain failure; downstream-effect absence then inconclusive follow. Full Discovery, Controlled Recovery, Figure 7 filtering, and summaries reuse this rule. | Aggregation precedence and Controlled F1 tests; dev edge-output gate | PASS |
| Robustness multiprocessing lifecycle | Every robustness fit could create and destroy a Windows process pool. | Development-shaped workload implied 60 pool lifecycles. | One reusable outer pool executes condition jobs; each bootstrap job uses one inner worker; nested pools are prohibited. Statistical settings are unchanged. | workers=1/workers=N equality and performance smoke | PASS |

## Method wording and unchanged contracts

Unrestricted temporal search removes structured candidate-edge constraints and
semantic branch constraints while retaining the same lag range, OLS
formulation, screening threshold, BH procedure, whole-trajectory bootstrap
repetitions, and support threshold. The hypothesis space and semantic FDR
grouping are therefore different, but the core temporal estimation and
bootstrap procedure are unchanged. `without_structured_representation` uses
this same definition.

The mechanism-disabled robustness table reports overall point-graph statistics
after one targeted mechanism is disabled. It tests whether the targeted pathway
may weaken or disappear; it does not require unrelated dynamics to vanish.

The formal configuration remains at 24 seeds, lags 1--5, parent alpha 0.10,
within-family BH-FDR 0.05, 100 trajectory bootstraps, support 0.65, 500 paired
bootstraps, 95% intervals, onset start 0, minimum standardised effect 0.10,
four consecutive steps, evaluation start 15, and lag tolerance 2. No Stage 2
estimator or threshold was changed.

## NON_SCIENTIFIC verification

- Pytest: 88 passed.
- Toy smoke: PASS, `smoke_runs/final_prerun_20260902_01/`.
- Two-scenario dev E2E: PASS,
  `dev_runs/final_prerun_20260902_01/`; every stage from semantic through render
  completed, the run is frozen, and `real_llm_api_called=false`.
- Dev final gates: mechanism alignment, distinct representation repetitions,
  n=1 missing stability, exact prospective paths, assortativity relabel
  invariance, contradiction precedence, Figure 4, Figure 7, and pool lifecycle
  all passed.
- Dev robustness profile: 60 outer jobs, 1 actual pool, 0 nested pools, 7.248 s
  sweep wall time.
- Lifecycle performance smoke: identical outputs; legacy 60 pools / 301.480 s,
  reused architecture 1 pool / 5.190 s.

## Evidence boundary

These results establish software and pre-run contract readiness only. They do
not provide formal scientific effect sizes, recovered relations, support rates,
contradiction rates, prospective outcomes, or paper figures.

## FORMAL RUN READINESS

**READY**

Unique formal command (run only after configuring the real local LLM key):

```powershell
E:\conda\python.exe run_experiment.py --config config\experiment.json --llm-config config\llm_api.local.json --run-id formal_acl2026_20260902_01 --workers auto --plot-repo "..\llm-agentic-dis"
```
