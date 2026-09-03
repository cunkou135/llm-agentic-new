# Path-centered protocol implementation report

## 1. Previous Stage 1

The previous contract asked one LLM response to define indicators, arbitrary
indicator groups, candidate relations, and prospective predictions. Stage 2
qualified relations, Stage 3 evaluated retained relations, and complete paths
were reconstructed afterward. That design could leave individually supported
relations without a prospectively coherent complete mechanism path.

## 2. New two-phase Stage 1

`indicator_generation` makes six Phase A calls per scenario. Each call can
return only executable indicators. The selected generation is chosen from the
first three by validator validity, public-source diversity, direct Micro
parameter coverage, Meso structural diversity, Macro concept diversity,
repair count, and a canonical hash tie-break. The final three calls are
replication-only.

`indicator_freeze` writes the selected 28 indicators and their SHA256.
`path_generation` then makes one primary plus two replication-only Phase B
calls over that exact frozen set. Phase B cannot add or edit indicators. The
first valid primary response supplies the formal paths and prospective
predictions; downstream numerical results never participate in selection.

## 3. Removal of arbitrary indicator grouping

The LLM schemas, prompts, frozen indicator payloads, full-discovery candidate
relations, temporal records, robustness operators, rendering interfaces, and
documentation no longer contain the former arbitrary grouping field. The only
statistical grouping is `hypothesis_group_id`, deterministically derived from a
frozen path's Macro endpoint before data exist.

## 4. Indicator Generation schema

`IndicatorGeneration` contains `scenario`, `phenomenon`, `indicators`, and
`interpretation_boundary`. Each `IndicatorSpec` contains its ID, scientific
semantics, scale/entity scope, public source fields, executable DSL AST,
temporal aggregation, parameter associations, and rationale. Pydantic forbids
all extra fields, so Phase A rejects candidate relations, candidate paths, and
prospective predictions.

## 5. Path Generation schema

`PathGeneration` contains `scenario`, `indicator_set_sha256`,
`candidate_paths`, and `prospective_predictions`. Each `CandidatePath` freezes
the real parameter and direction, exact Micro/Meso/Macro IDs, two relation
directions, three expected responses, rationale, mechanism explanation, and a
falsification condition. Each `ProspectivePrediction` references an existing
`candidate_path_id`.

## 6. CandidatePath capacity contract

Every scenario must contain 16--24 unique indicator triples. All three
controllable parameters need at least four paths. Every one of the four frozen
Macro indicators needs at least two paths. Scale, direct Micro association,
public-field executability, genuine Meso organization, adjacent-scale
non-triviality, and Micro--Macro lineage checks fail closed.

## 7. Path projection to unique relations

For every frozen path, the program projects Micro-to-Meso and Meso-to-Macro.
It deduplicates exact source/target pairs, records every contributing path ID,
and attaches every relevant Macro-outcome group. Conflicting prospective
directions are preserved as `mixed`; they are not silently resolved.

## 8. FDR hypothesis groups

The group is `macro_outcome_<macro_indicator_id>`. A relation used by paths to
multiple Macro endpoints is fitted once but participates separately in each
predefined multiple-testing family, producing group-specific q-values. No
group is adjusted from observed p/q values.

## 9. Stage 2 complete-path decision

Stage 2 retains the unchanged standardized lagged OLS, target self-history,
competing parents, lag 1--5, parent alpha 0.10, BH/FDR 0.05, whole-trajectory
bootstrap 100, and support 0.65. A frozen path is temporally qualified only if
both adjacent relations are retained inside that path's Macro-outcome group.
All candidate paths, including failures, are written to
`path_temporal_qualification.csv`.

## 10. Stage 3 complete-path decision

A temporally qualified path is `supported` only when the true parameter
manipulation succeeds, required Micro/Meso/Macro responses are significant,
both adjacent intervention classifications are supported for the same
scenario/parameter/direction/root, frozen directions agree, onset ordering
satisfies the original rule, and observational lag consistency holds. A clear
directional contradiction yields `contradicted`; a failed root manipulation
yields `manipulation_failure`; remaining insufficient evidence is
`inconclusive`.

## 11. Prospective binding

Six predictions per scenario are emitted in Phase B, each keyed by a frozen
`candidate_path_id`, and hashed before baseline simulation. The later
`prospective` stage evaluates those immutable predictions; it cannot invent a
replacement path.

## 12. Holdout confirmation

Holdout uses only independent seeds and frozen primary paths. It does not
regenerate semantics, select relations/lags/thresholds, or replace a failed
path. Confirmation output preserves `primary_result_unchanged=true`.

## 13. New confirmatory seed pool

Primary seeds are 3101--3124 and holdout seeds are 4101--4112. They are fixed,
disjoint, and validated at config load. The old 1101--1124 and 2101--2112 pools
are treated as development evidence and are not reused for the new
confirmatory run.

## 14. Formal simulator task count

The formal task matrix remains 864 trajectories: 672 primary plus 192
holdout. The protocol increases path-hypothesis richness without increasing
simulator workload.

## 15. Automated tests

The repository test suite contains 175 passing tests. It covers Phase A
forbidden outputs and scale rules; freeze/hash failures; exact Phase B
references, capacity, coverage and duplicate rejection; deterministic relation
projection; group-specific path qualification; strict path intervention
classes; prospective binding; holdout immutability; exact-ID replication; seed
and 864-task contracts; and all unchanged Stage 2/3 threshold gates.

## 16. Toy smoke

`smoke_runs/path_protocol_v4_20260903` completed successfully with deterministic
toy data. It is explicitly marked non-scientific and is not evidence for any
paper claim.

## 17. Development E2E

The inexpensive deterministic mock run
`dev_runs/path_protocol_v4_split_20260903` passed all 19 stages and its final
contract audit. It verified required files, immutable references, path outputs,
holdout non-replacement, exports, and rendering. It produced zero supported
paths; this was preserved as a non-scientific dev outcome and correctly did not
trigger protocol adjustment. Dev acceptance does not require a supported path.

## 18. Unresolved scientific correctness blockers

No unresolved implementation-level scientific-contract blocker is known. This
does not assert that any path is scientifically supported: no new formal run
has been executed, so scientific outcomes remain unknown until the user runs
the frozen confirmatory protocol.

## Final invariants

- Simulator core unchanged.
- Stage 2 model and thresholds unchanged.
- Stage 3 thresholds unchanged.
- Onset definition unchanged.
- `res_f` preserved as development evidence.
- No hidden truth used in Full Discovery indicator/path/prediction generation
  or numerical discovery.
- No real LLM API called during implementation.
- No formal scientific experiment executed during implementation.
