# Frozen path-centered experiment plan

## Evidence flow

1. Generate six independent 16/8/4 executable indicator sets using public
   simulator semantics, parameters, raw schemas, scale rules, and the safe DSL.
2. Select only among the first three by a data-blind rule and immediately
   freeze/hash the selected indicators; retain the other three for replication.
3. From the frozen IDs, generate one primary and two replication-only sets of
   16--24 complete parameter-to-Micro-to-Meso-to-Macro paths. Freeze the first
   valid primary set and its six path-bound prospective predictions.
4. Project unique adjacent-scale relations and assign pre-data Macro-outcome
   hypothesis groups.
5. Generate public and isolated hidden baseline trajectories.
6. Apply unchanged lagged OLS, group-local BH/FDR and whole-trajectory
   bootstrap; qualify only paths whose two group-specific relations survive.
7. Generate matched-seed primary intervention trajectories, preserving minus
   and plus as the only primary support conditions.
8. Classify each frozen path under the strict manipulation, response,
   direction, onset, lag and two-relation requirements.
9. Freeze primary results, run secondary five-point dose analysis, and confirm
   the same paths on independent holdout seeds.
10. Run circular-shift path negative controls, observation/representation
    robustness, data efficiency, Controlled Recovery, export, and render.

## Isolation guarantees

```text
public schema -> Phase A indicators -> freeze -> Phase B paths -> Full Discovery
withheld states ----------------------------------------------> Controlled Recovery only
```

Generated indicators are never aligned to hidden identities. Hidden truth
scores exist only for fixed Controlled Recovery processes. Smoke and dev output
trees are non-scientific and excluded from source/formal artifact manifests.

## Formal configuration

- primary seeds 3101--3124; holdout seeds 4101--4112;
- 864 total simulator trajectories;
- lag 1--5, parent screen 0.10, Macro-group FDR 0.05;
- 100 trajectory bootstraps, support 0.65;
- 500 paired-seed intervention bootstraps and 95% CI;
- standardized threshold 0.10, onset start 0 and four consecutive steps;
- evaluation start 15, terminal window 24, lag tolerance 2;
- automatic workers capped at 12;
- render settings frozen in `config/experiment.json`.

All semantic and stage artifacts are immutable and hashed. Any mismatch or any
attempt to rerun semantics after baseline starts fails closed and requires a new
run ID.
