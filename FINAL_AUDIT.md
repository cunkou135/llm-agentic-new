# Final pre-run scientific-logic audit

Audit date: 2026-09-02

This audit covers the current source tree after the final scientific-logic
remediation. No formal 24-seed experiment and no real LLM API call were made.
The toy and development outputs cited below are explicitly NON_SCIENTIFIC.

## Required correction matrix

| Gate | Implemented contract | Verification evidence | Status |
|---|---|---|---|
| Stage 3 path-aware propagation | A Micro->Meso retained edge uses a direct Micro parameter root. A Meso->Macro edge resolves a same-branch upstream Micro root and reuses its simulator parameter. Only that parameter is changed; root, edge source, and edge target are observed from the same matched-seed trajectories. Output records `root_source`, `edge_source`, `edge_target`, `manipulation_level`, and `intervention_scope` (`direct_root` or `upstream_mediated`). | Direct-root, upstream-mediated, no-automatic-unmapped, onset-order, and end-to-end coverage checks | PASS |
| Controlled intervention denominator | Intervention recall is `supported eligible truth / eligible truth`. Truth relations without a legal simulator manipulation route are outside the estimand (`not_applicable`) rather than false negatives. Outputs include eligible/supported truth counts and intervention precision, recall, and F1. | Both dev scenarios report 6 eligible truth edges for Full Method; denominator unit test passes | PASS |
| Controlled direct manipulation semantics | Schelling `destination_preference` is mapped to `s_micro_destination_similarity = mean(destination_similarity)`, the public observable directly adjacent to destination selection. It is no longer mapped to mean current-location similarity. Other direct mappings remain rule-level gates or update magnitudes. | Rule-consistency unit test and public-field inspection | PASS |
| Contradiction vocabulary | The only Stage 3 directional contradiction state is `directionally_contradicted`. `contradicted` is rejected as a legacy Stage 3 value. Contradiction rate excludes `not_applicable` but includes `manipulation_failure` in the applicable-attempt denominator. | Vocabulary/rate unit test; dev classification scan contains 0 legacy rows | PASS |
| Candidate denominator | Structured, single-trajectory, vote, and semantic proposal rows use the structured candidate count. Unrestricted temporal search uses the actual ordered-pair search space. A rate above one raises an error. | Both 28-node dev scenarios report unrestricted count 756; maximum qualification rate is 1.0 | PASS |
| Prospective falsification | After successful source manipulation, any missing required downstream response, wrong required direction, temporal-order violation, or missing required observational edge is `contradicted`. Only all required evidence passing is `supported`; required absence cannot become partial support. | Missing-Meso, missing-Macro, wrong-direction, and all-pass tests | PASS |
| Evidence integration | `analysis/attribution_objects.json` reads temporal and intervention evidence only from `full_method`. Non-primary method intervention rows are exported to `analysis/comparative_method_intervention_evidence.csv`. | Dev attribution method set is exactly `full_method`; isolation unit test passes | PASS |
| Controlled branch-local FDR | Controlled nodes, truth candidates, and distractors use exactly four explicit families: `controlled_branch_0` through `controlled_branch_3`. | Both controlled representations expose four unique branch IDs; branch-local tests pass | PASS |
| Semantic resume | Each generation writes immutable prompt/request/response history, a checkpoint containing artifact and accepted-payload hashes, and a `generation_result.sha256` sidecar. Resume verifies all hashes and the current semantic contract before skipping. Completed generations and repair history cannot be silently overwritten. | Simulated two-complete/third-fails/resume test; 4/4 dev results have checkpoint metadata and sidecars | PASS |
| Figure 7 propagation filter | Eligible paths require significant Micro, Meso, and Macro responses; ordered onsets; and `supported` Full Method intervention classifications for both Micro->Meso and Meso->Macro edges. Macro magnitude ranks only paths that pass these gates. | Incomplete/unordered path test passes; dev rendering honestly displays no Full Discovery path rather than plotting unsupported propagation | PASS |

## Frozen method settings

The formal configuration remains unchanged:

- 16 Micro, 8 Meso, and 4 Macro generated observables;
- 28--48 structured candidate edges;
- level-based lagged OLS with lags 1--5, target self-history, and competing parents;
- parent screen 0.10 and within-branch BH-FDR 0.05;
- 100 whole-trajectory bootstrap repetitions and support threshold 0.65;
- actual simulator parameter interventions with 24 matched seeds;
- minus/baseline/plus conditions, 500 paired bootstrap repetitions, and 95% intervals;
- onset start 0, minimum standardised effect 0.10, four consecutive steps;
- evaluation start 15 and lag tolerance 2.

No PCMCI+, DYNOTEARS, packaged Granger method, Pearson substitute,
difference-only model, post-result threshold change, or direct manipulation of a
generated indicator was introduced.

## Verification results

- Pytest: **66 passed** using `E:\conda\python.exe -m pytest -q`.
- Toy smoke: **passed**, `scientific_evidence=false`, output
  `smoke_runs/final_logic_20260902_02/`.
- NON_SCIENTIFIC development E2E: **passed**, output
  `dev_runs/final_logic_20260902_02/`.
- Development scope: Schelling and Deffuant, deterministic mock semantic
  responses, isolated baseline/intervention simulations, temporal analysis,
  paired effects, prospective validation, robustness, export, Figure 2--8, and
  final run freeze.
- Stage 3 development audit: 6 Micro->Meso testable edges, 7 Meso->Macro
  testable edges, 3 supported Micro->Meso edges, 3 supported Meso->Macro edges,
  5 complete testable paths, and 3 complete supported paths in Controlled
  Recovery. Both scenarios contribute Meso->Macro and complete-path checks.
- Full Discovery also exercised upstream-mediated classification: 3 testable
  Meso->Macro retained edges and 2 complete testable paths. Unsupported paths
  were not promoted to Figure 7.
- Controlled Full Method eligible-truth denominator: **6 per scenario**.
- Full Discovery unrestricted candidate denominator: **756 per scenario**.
- Main attribution intervention method set: **`full_method` only**.
- Legacy Stage 3 `contradicted` rows: **0**.
- Development run is frozen and marked `real_llm_api_called=false`.
- `git diff --check`: **PASS**; line-ending notices are informational only.

## Evidence boundary

Passing this audit establishes code-path readiness and repeatable non-scientific
execution. It does not establish the eventual formal scientific effect sizes,
support rates, recovery scores, contradictions, prospective outcomes, or
Figure 7 paths. Those conclusions must come only from the new formal 24-seed
run with real frozen LLM generations.

## FORMAL RUN READINESS

**READY**
