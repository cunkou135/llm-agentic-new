# Final pre-run audit

Audit date: 2026-09-02

Scope: current project only. No real LLM API, formal 24-seed experiment,
formal 100-repetition trajectory bootstrap suite, or formal 500-repetition
paired bootstrap suite was run. All cited executions are NON_SCIENTIFIC.

This audit includes the pre-formal scale-protocol revision. The prior `res1`
release remains development/pilot evidence for the older permissive scale
contract and is not eligible for resume or scientific-artifact reuse.

## Scale-protocol upgrade gates

| Gate | Verified contract | Status |
|---|---|---|
| Schelling structural Meso | Fixed 3 x 3 public districts on the unchanged periodic formal grid; district membership is a primitive, not an outcome; relocation remains global and stochastic. | PASS |
| Deffuant structural Meso | Dynamic simple undirected network; rejected/backfire encounters use fixed-probability adaptive rewiring; edge count and non-isolation are preserved; `disable_backfire` retains rewiring. | PASS |
| Typed scientific scale | Micro entity scopes are elementary/local; Meso scopes require a real group/network structural operator; Macro scope is whole system. | PASS |
| Trivial lineage rejection | Nested smoothing, differencing, clipping, and constant-rescaling-only cross-scale edges fail with `trivial_cross_scale_transform`. | PASS |
| Public/hidden boundary | District/network primitives are public; hidden channels remain separately stored; Full Discovery inference modules do not import reference truth. | PASS |
| Counter-based pairing | Same seed reproduces bitwise trajectories and aligned initial states; new rewiring draws use independent time/event/agent keys. | PASS |
| Empty scientific result handling | Intervention-only absent-event indicators preserve NaN as inconclusive while baseline all-NaN fails; zero retained paths preserve a headered empty schema through export/render. | PASS |
| Frozen formal settings | 16/8/4, four branches, 28--48 edges, three generations/repairs, 24 seeds, and all Stage 2/3 thresholds are checked at formal config load. | PASS |

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
| Condition- and root-specific propagation validation | Path validation could borrow edge support from another parameter, intervention direction, or direct Micro root. | Evidence was collapsed across `parameter`, `direction`, and `root_source` before checking a path. | Each path now requires both edges to be supported under its own `scenario + parameter + direction + root_source + full_method` subset. | Cross-direction and cross-root evidence-isolation regression tests | PASS |
| Zero baseline variance | A zero or invalid baseline SD was replaced by 1, creating an artificial standardised effect; later Stage 3 sign conversion could also crash on the resulting NaN. | Non-estimable indicators received a finite standardised effect, or `int(np.sign(NaN))` could raise. | Raw effects remain available; standardised effects and intervals are NaN, significance is false, onset/peak are -1, and Stage 3 classifies non-finite effects as inconclusive before sign conversion. | Mixed estimable/non-estimable effect and no-crash classification regression tests | PASS |
| Semantic-proposal temporal metric | The unfiltered LLM candidate graph produced a mechanical qualification rate of 1. | The semantic proposal could be misread as 100% temporally qualified despite having no temporal test. | `llm_semantic_proposal` now reports a missing qualification rate with reason `not_temporally_qualified`; temporally evaluated methods retain the candidate-retention calculation. | Semantic-proposal metric-scope regression test | PASS |
| Hidden-reference robustness serialization | Full Discovery mechanism-disabled output included controlled-reference edge identities. | `mechanism_disabled_checks.csv` serialized `targeted_reference_edges` even though they were not used in fitting. | The CSV retains public mechanism-variant context and overall point-graph metrics but no longer imports or serializes hidden reference edges. | Mechanism-check output-schema regression test | PASS |
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
four consecutive steps, evaluation start 15, terminal window 24, and lag tolerance 2. No Stage 2
estimator or threshold was changed.

## NON_SCIENTIFIC verification

- Pytest: 111 passed.
- Toy smoke: PASS, `smoke_runs/protocol_upgrade_20260902_01/`.
- Two-scenario dev E2E: PASS,
  `dev_runs/protocol_upgrade_20260902_04/`; every stage from semantic through render
  completed, the run is frozen, and `real_llm_api_called=false`.
- Dev final gates: mechanism alignment, distinct representation repetitions,
  n=1 missing stability, exact prospective paths, assortativity relabel
  invariance, contradiction precedence, Figure 4, Figure 7, and pool lifecycle
  all passed.
- Dev robustness profile: 60 outer jobs, 1 actual pool, 0 nested pools, 5.660 s
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
E:\conda\python.exe run_experiment.py --config config\experiment.json --llm-config config\llm_api.local.json --run-id formal_scale_v2_001 --workers auto --plot-repo "..\llm-agentic-dis"
```
