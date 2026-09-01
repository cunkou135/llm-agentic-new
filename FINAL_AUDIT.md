# Final pre-run audit

Readiness: **READY**

This is a software and scientific-contract readiness decision. No formal experiment and no real model API call was executed during remediation.

| Issue | Old behavior | New behavior | Affected artifacts | Test or evidence | Status |
|---|---|---|---|---|---|
| Public/reference isolation | Withheld mechanism state appeared in the public schema and public simulator payload | Public schemas contain only documented simulator logs; withheld states use separate payloads, paths, hashes, and sidecars | `raw_schemas.py`, `simulators.py`, `simulation.py`, `controlled.py` | Public-schema, prompt, generated-field, NPZ-overlap, and compiler-isolation tests | PASS |
| Formal NPZ resume | Existing NPZ could acquire a new sidecar during resume | Existing formal NPZ without its original sidecar fails closed | `simulation.py` | `test_formal_npz_without_checkpoint_fails_closed` | PASS |
| Evaluation meaning | Generated observables could be exact-signature aligned to hidden identities | Full Discovery has no hidden alignment or hidden-reference recovery score; Controlled Recovery uses fixed benchmark identities only | `evaluation.py`, `controlled.py` | Full Discovery reference columns are all missing with reason; both track files present | PASS |
| Stage order | Simulation preceded semantic freeze and simulation phases were combined | Nine fixed checkpoints freeze representation and predictions before baseline data and separate intervention simulation from intervention analysis | `pipeline.py`, `progress.py`, run manifest | Stage-order test; all nine dev checkpoints completed | PASS |
| Temporal aggregation | Aggregation was an unchecked dictionary until execution | Typed four-operation contract with strict positive rolling window and no extra fields | `schemas.py`, `dsl.py`, `semantic.py` | Illegal aggregation tests | PASS |
| DSL documentation and entropy | Operator grammar lacked per-operation contracts; continuous entropy was accepted | Every operator has required/optional/type/axis/output/example fields; continuous data require binned entropy | `dsl.py` | Grammar and entropy tests | PASS |
| Parameter and prospective paths | Parameter presence did not guarantee a legal manipulation source | Every parameter requires a direct micro source; prediction source and ordered candidate path are validated; machine-checkable criteria are frozen | `schemas.py`, `semantic.py`, `prospective.py` | Direct-source, source-match, real-path, and freeze-order tests | PASS |
| Representation participation | A valid branch could still contain isolated indicators | All 16 micro, 8 meso, and 4 macro nodes must participate; edge range is frozen at 28--48 | `semantic.py`, both experiment configs | Node-participation and edge-bound tests | PASS |
| Unrestricted comparison | Unrestricted search used only a point estimate | Full and unrestricted paths use the same OLS, FDR, bootstrap repetitions, and support threshold; candidate space alone differs | `temporal.py`, `robustness.py` | Same-bootstrap-contract test | PASS |
| Stability semantics | Point graphs used artificial support 1, lag support 1, and lag variance 0 | Point and single-trajectory records use missing stability fields; bootstrap/vote alone report empirical stability | `temporal.py` | Point and single-trajectory stability tests | PASS |
| Functional ablations | Joint-trajectory and unstructured variants did not match the manuscript definitions | Joint-trajectory removal uses one trajectory; structured-representation removal uses the same 28 generated observables with unrestricted bootstrap search | `robustness.py` | Source inspection plus method-contract tests | PASS |
| Predefined comparator | Meso/macro expressions read withheld channels | All 28 fixed comparator expressions use public simulator fields only | `predefined.py` | Public-field comparator test for both scenarios | PASS |
| Intervention coverage | Only the full graph was classified | Full, vote, single, and unrestricted graphs use the same paired effects; empty graphs receive missing metrics plus an explicit reason | `interventions.py`, `controlled.py`, `evaluation.py` | All-method test and final dev method tables | PASS |
| Runtime fairness | Per-method runtime and fair speedups were incomplete | Semantic generation is a separate runtime scope; four temporal methods have comparable analysis timing; speedups use analysis rows only | `temporal.py`, `provenance.py` | Runtime-export test and final dev runtime table | PASS |
| Provenance ownership | Source manifests included non-scientific outputs and temporal owned later-mutated result files | Formal/dev/smoke trees are separated; non-scientific trees are excluded; each checkpoint owns only immutable outputs; required timestamps are recorded | `provenance.py`, `pipeline.py` | Source-manifest, resume, hash, and timestamp checks | PASS |
| Progress totals | Stage completion could finish at the initial placeholder total | Actual callback total becomes the durable stage total and completion reaches that total | `progress.py` | Final dev progress completed all nine stages | PASS |
| Worker policy | Automatic workers could expand to nearly every logical CPU | Automatic workers are capped at `min(cpu-1, 12)`; explicit 1/N behavior remains deterministic | `run_experiment.py`, `temporal.py` | Single/two-worker equality test | PASS |
| Figure 2--8 | Renderer departed from the original visual language and contained an opacity hardcode and quartile band | Dynamic renderer migrates the original layout/palette, reads scientific parameters only from the frozen config, displays false edges, uses mean effects, full effect range, and median nested 95% endpoints | `rendering.py`, `config/experiment.json` | Render-hardcode test; seven final dev PNGs; visual QA | PASS |
| Documentation | Run order and artifact map described the superseded pipeline | Runbook gives the fixed order, split artifacts, resume behavior, and one recommended formal command | `RUNBOOK.md`, `README.md`, implementation documents | Manual audit | PASS |
| Prohibited name | Required case-insensitive zero-match gate was not part of final evidence | Repository source scan excludes generated/formal output trees and returns zero matches | Entire source tree | `test_forbidden_name_zero_matches`; independent scan count 0 | PASS |

## Verification results

- Pytest: **48 passed in 10.28 s**.
- Toy smoke: **passed**, 8 bootstrap repetitions, `scientific_evidence=false`, output `smoke_runs/final_smoke_20260902b/`.
- Development end to end: **passed**, output `dev_runs/final_dev_20260902i/`.
- Development scope: both real simulator paths, 2 seeds, small agents/steps, 6 temporal bootstraps, 6 intervention bootstraps, deterministic mock semantic responses, all nine stages, resume, and Figure 2--8.
- Development provenance: `formal_output=false`, `real_llm_api_called=false`, `RUN_FROZEN` present, 0 artifact hash mismatches, 0 hidden fields in public NPZ files.
- Development outputs: Full Discovery and Controlled Recovery tables, all five runtime methods, all four intervention graph methods, and seven rendered figures.
- Forbidden-name scan: **0 matches**.
- `git diff --check`: **PASS** after end-of-file cleanup.

## Remaining blockers

None for starting the formal run. A valid local API configuration and a fresh run identifier are operational inputs, not evidence of a completed formal experiment.

## Sole formal command

```powershell
.\.venv\Scripts\python run_experiment.py --config config\experiment.json --llm-config config\llm_api.local.json --run-id rerun_001 --workers auto --plot-repo "..\llm-agentic-dis"
```
