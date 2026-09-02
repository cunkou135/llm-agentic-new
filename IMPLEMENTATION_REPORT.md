# Pre-formal scale-protocol upgrade implementation report

Date: 2026-09-02

Status: implementation and non-scientific development verification complete.
No real LLM API and no formal 24-seed experiment were executed.

## 1. Method unchanged

The existing project was upgraded in place. The paper's three-stage method and
prospective protocol remain unchanged:

```text
semantic -> baseline_simulation -> temporal -> intervention_simulation
         -> intervention -> prospective -> robustness -> export -> render
```

Stage 2 still uses level-based standardized lagged OLS, target self-history,
competing candidate parents, lags 1--5, parent alpha 0.10, branch-local BH/FDR
0.05, 100 whole-trajectory bootstraps, and support 0.65. Stage 3 still uses
parameter-only matched interventions, 500 paired bootstraps, 95% intervals,
effect threshold 0.10, four consecutive steps, evaluation start 15, terminal
window 24, and lag tolerance 2. Formal configuration validation now explicitly
fails if any of these frozen values, the 24 seeds, or the representation
capacity settings are changed.

The 16 Micro / 8 Meso / 4 Macro, four-branch, 28--48-edge contract with three
independent generations and three repair rounds is documented as this
experiment's representation capacity control, not as a universal theoretical
requirement of the method.

## 2. Modified files

- Protocol/configuration: `config/experiment.json`,
  `config/dev_experiment.json`, `config/semantic_prompt.txt`.
- Simulator and public schema: `src/emergence_attribution/simulators.py`,
  `src/emergence_attribution/raw_schemas.py`.
- Safe DSL and semantic contract: `src/emergence_attribution/dsl.py`,
  `src/emergence_attribution/schemas.py`,
  `src/emergence_attribution/semantic.py`,
  `src/emergence_attribution/predefined.py`.
- Controlled Recovery and orchestration safeguards:
  `src/emergence_attribution/reference_truth.py`,
  `src/emergence_attribution/pipeline.py`,
  `src/emergence_attribution/simulation.py`,
  `src/emergence_attribution/interventions.py`.
- Tests: `tests/test_dsl_and_schema.py`,
  `tests/test_final_remediation.py`, `tests/test_protocol_upgrade.py`.
- Documentation: `README.md`, `RUNBOOK.md`, `IMPLEMENTATION_PLAN.md`,
  `IMPLEMENTATION_REPORT.md`, `FINAL_AUDIT.md`.

## 3. Schelling scenario upgrade

The periodic grid, two fixed groups, dissatisfaction, relocation, destination
preference, and the three existing intervention parameters are retained. The
30 x 30 formal grid is partitioned deterministically into 3 x 3 fixed spatial
districts. Development grids use the same integer partition rule, including
uneven dimensions. Relocation still samples vacancies over the whole periodic
grid; a district is therefore an environment primitive, not a confinement rule
or precomputed segregation answer.

At the start of each recorded step, every agent's current position is mapped to
its fixed cell district. Moore-neighborhood calculations remain periodic.
`disable_homophilic_relocation` still removes only preferential destination
selection.

## 4. Deffuant scenario upgrade

The initial Watts--Strogatz network, bounded-confidence assimilation,
rejection, backfire, and the three existing intervention parameters are
retained. Two public fixed mechanism constants are centralized in scenario
configuration:

- adaptive rewiring probability: 0.15;
- weak homophilic choice probability: 0.65.

After a rejected or backfire encounter, an independent counter-keyed draw may
replace the sampled tie. Candidates exclude the focal node, old partner, and
current neighbors. A small deterministic random-priority candidate sample is
used; the weak-homophily draw selects the closest-opinion candidate within that
sample. Every step preserves a simple undirected graph, constant edge count,
and at least one tie per agent. The saved edge list is the network used at the
start of that step; successful rewiring affects the next step.

`disable_backfire` sets only repulsive opinion-update strength to zero. Distant
encounters remain rejected and continue to use the adaptive rewiring rule.

## 5. Public raw-schema additions

Schelling adds primitive `agent_id[agent]` and
`district_id[time,agent]`; existing `agent_position`, `agent_group`, local event
fields, and `state_grid` remain available. No district entropy, segregation,
turnover score, or Meso label is logged.

Deffuant changes `network_edges` from a static edge list to
`network_edges[time,edge,endpoint]` and adds
`edge_rewired[time,agent]`. The existing opinion, partner, distance,
acceptance/rejection/backfire, and shift fields remain. No community ID,
modularity, echo-chamber score, polarization cluster, or hidden truth is logged.

## 6. Generic DSL additions

The safe declarative DSL adds:

- `rolling_std`;
- `group_reduce` with mean, sum, count, fraction, variance, standard deviation,
  or categorical entropy;
- `network_neighborhood_reduce` with generic local reducers;
- dynamic `network_assortativity` and `network_density` support;
- `network_component_count`;
- `network_largest_component_fraction`.

AST type/dimension validation precedes execution. Dynamic edge slices reject
non-finite/out-of-range endpoints, self-loops, and duplicate undirected edges.
No-neighbor or absent-event quantities have explicit NaN behavior. An
all-undefined baseline indicator still fails closed; an intervention that
legitimately removes every conditioned event preserves NaN for Stage 3 to mark
inconclusive. A zero-row path result is written with a stable CSV schema rather
than as a malformed headerless file.

## 7. Stage 1 scale contract

`IndicatorSpec` now contains a machine-readable `entity_scope`.

- Micro accepts `individual`, `interaction`, `elementary_event`, or
  `local_process` and must derive from a public agent/interaction/cell/edge
  primitive. Population prevalence of an elementary event remains valid Micro.
- Meso accepts `neighborhood`, `district`, `community`, `cluster`, or
  `local_domain` and must actually use a group/network structural operation.
- Macro requires `whole_system`.

The prompt explains these scientific entities without supplying fixed
observable names. The LLM still selects semantics, computations, branches,
candidate edges, parameter associations, and prospective predictions.

## 8. Trivial cross-scale transform rule

Each cross-scale candidate edge is compared through a canonical computational
lineage. Identity/temporal aggregation, rolling mean/std, time difference,
negation, square-root/log transformations, clipping, and constant
add/subtract/multiply/divide/safe-ratio wrappers are stripped. If source and
target then have the same core computation and the target introduces no new
group/network/spatial structural operator, validation rejects the edge with:

```text
trivial_cross_scale_transform
```

Thus nested rolling windows, differencing, and constant rescaling cannot form a
Micro -> Meso -> Macro mechanism. Rolling operations remain legal inside a
scientifically distinct whole-system indicator.

## 9. Controlled Recovery update

The fixed hidden-known benchmark was synchronized with the simulator revision.
Schelling hidden channels are anchored to district-level heterogeneity and
whole-grid spatial contexts. Deffuant hidden channels are anchored to dynamic
network-neighborhood statistics and now include an adaptive-rewiring Micro
process/branch. Guaranteed delayed Micro -> Meso -> Macro channels remain
separately labelled Controlled Recovery; they are not generated-node answers.

## 10. Hidden-truth isolation

Semantic generation receives `public_raw_schema()` only and contains an
explicit hidden-field denylist. Public and hidden NPZ payloads are written to
separate directories with separate hashes. Full Discovery semantic, temporal,
and intervention modules do not import `reference_truth` and do not read
`mechanism_channel`. Only the controlled module combines public primitives with
the hidden-known channels. Generated nodes are not aligned to truth; Full
Discovery F1/SHD/lag-MAE remain N/A.

## 11. Matched-seed RNG contract

Initial states and initial networks are functions only of scenario and seed.
Existing partner/activation streams retain their keys. Adaptive rewiring uses
independent streams for trigger, homophily choice, and per-agent candidate
priority (`seed + time + event stream + agent`). Because each draw is produced
by a counter-based generator, a missing assimilation/backfire event does not
shift later global RNG state. Regression tests verify bitwise reruns, equal
initial states across minus/baseline/plus, aligned time-zero partner streams,
preserved rewiring under `disable_backfire`, and stable public/hidden payload
digests.

## 12. Non-scientific verification

- Pytest: **111 passed**.
- Toy smoke: **PASS** at
  `smoke_runs/protocol_upgrade_20260902_01/`; the report states
  `scientific_evidence=false`.
- Final two-scenario dev E2E: **PASS** at
  `dev_runs/protocol_upgrade_20260902_04/`.
- All nine stages completed, the dev run is frozen, 245 artifacts were hashed,
  seven PNG figures rendered, `formal_output=false`, and
  `real_llm_api_called=false`.
- Semantic and prediction freeze timestamps precede baseline simulation start.
- The small mock Full Discovery run legitimately produced no complete supported
  path; it was not treated as a failure and no threshold was relaxed. These dev
  counts are not paper evidence.

## 13. Academic handling and remaining blockers

The prior `res1` release remains development/pilot evidence showing that the
earlier scale contract permitted mathematically nested observables to occupy
different scales. It was not deleted, resumed, or used as scientific input to
this implementation. The next formal run must begin with semantic generation,
use a fresh run id, and never combine old and new scientific artifacts.

No unresolved scientific-correctness blocker was found in the upgraded code
under the permitted tests. This means the implementation is ready to start a
new formal run; it does **not** mean the revised method is guaranteed to recover
or intervention-support any path. That remains a formal experimental result.

No real LLM API, formal simulation, formal 100-bootstrap Stage 2 analysis,
formal 500-bootstrap Stage 3 analysis, or new paper result was produced during
this upgrade.
