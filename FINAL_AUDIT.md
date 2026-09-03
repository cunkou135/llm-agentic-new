# Final protocol audit

| Gate | Verification | Status |
|---|---|---|
| Phase A scope | Strict `IndicatorGeneration` rejects relationships, paths, predictions, and unknown fields; exact 16/8/4 validator retained. | PASS |
| Phase separation | `indicator_generation`, `indicator_freeze`, and `path_generation` are distinct checkpointed pipeline stages. | PASS |
| Indicator immutability | Frozen JSON and SHA256 are immutable; mismatch or post-baseline semantic execution fails closed. | PASS |
| Phase B references | Indicator hash, exact IDs, scales, real parameters, and direct Micro associations are validated. | PASS |
| Path capacity | 16--24 unique triples, >=4 paths/parameter, >=2 paths/Macro. | PASS |
| Relation derivation | Adjacent relations are deterministic, globally deduplicated projections of frozen paths. | PASS |
| Multiple testing | Macro-endpoint hypothesis groups are fixed before data; shared fits receive group-specific q-values. | PASS |
| Stage 2 path evidence | Both group-specific adjacent relations must be retained; failures remain in output. | PASS |
| Stage 3 path evidence | Same scenario, parameter, direction and Micro root; manipulation, response, direction, onset, lag and both relations required. | PASS |
| Zero baseline SD | Standardized effect NaN, significant false, onset -1; raw effect retained; classification does not crash. | PASS |
| Prospective freeze | Six predictions/scenario reference frozen path IDs and are hashed before baseline. | PASS |
| Holdout isolation | Independent seeds confirm only frozen primary paths and cannot change primary classification. | PASS |
| Secondary protocols | Five-point dose, path dose summary, circular-shift path negative control, robustness and data efficiency retained. | PASS |
| Representative path | Selected only from frozen supported paths by deterministic confirmation then lexical rule; otherwise null. | PASS |
| Rendering | Figure 5/7 accept only supported frozen path IDs and cannot reconstruct mixed evidence. | PASS |
| Hidden truth boundary | Hidden payload remains confined to Controlled Recovery; Full Discovery has no truth matching metrics. | PASS |
| Seeds/tasks | 3101--3124 primary, 4101--4112 holdout, disjoint, 864 formal trajectories. | PASS |
| Threshold freeze | Formal config enforces unchanged Stage 2, Stage 3 and onset settings. | PASS |
| Tests | `pytest -q`: 175 passed. | PASS |
| Toy smoke | Deterministic non-scientific smoke completed. | PASS |
| Real API/formal run | Neither was executed during implementation. | PASS |

The audit establishes software and protocol compliance only. It makes no claim
that a scientific path will be supported in the future confirmatory run.
