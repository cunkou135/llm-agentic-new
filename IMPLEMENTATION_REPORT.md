# Final pre-run implementation report

## Scientific architecture

The existing project was remediated in place. The formal chain is now:

```text
semantic -> baseline_simulation -> temporal -> intervention_simulation -> intervention -> prospective -> robustness -> export -> render
```

The selected semantic representation and structured prospective criteria are frozen and hashed before baseline simulation. Baseline and intervention simulations have distinct task manifests and stage checkpoints.

## Public and hidden data boundary

- Semantic generation and Full Discovery receive only public simulator fields.
- Public NPZ files are checked against a hidden-field deny-list.
- Withheld benchmark states are stored only under `data/reference_hidden/`, with separate hashes and sidecars.
- Formal existing NPZ files without their original sidecars fail closed.
- Full Discovery compiles indicators only from public NPZ files.
- Controlled Recovery is implemented in a dedicated module and is the only analysis path that combines public and hidden inputs.

## Evaluation tracks

`analysis/full_discovery_results.csv` reports executability, graph coverage, temporal qualification, bootstrap stability, lag stability, intervention support/contradiction, prospective validation, robustness, and runtime. Hidden-reference Edge F1, SHD, lag MAE, and direction accuracy are always missing with an explicit reason.

`analysis/controlled_recovery_results.csv` reports Edge F1, SHD, lag MAE, direction accuracy, stability, intervention F1, uncertainty width, and runtime for fixed hidden-known benchmark processes. `analysis/main_results.csv` combines both with `evaluation_track` labels.

## Method contracts

- The 16/8/4 representation has 28 participating nodes and a frozen 28--48 candidate-edge range.
- Every controllable parameter requires a direct micro-level source.
- Prospective sources must be direct micro associations and must follow real candidate-graph paths.
- Temporal aggregation is a strict typed contract.
- Continuous entropy requires explicit binning.
- Full and unrestricted searches use identical OLS, FDR, trajectory-bootstrap, and support settings; only candidate space differs.
- Point and single-trajectory graph records do not fabricate support, lag support, or lag variance.
- The `without_joint_trajectories` ablation uses the single-trajectory method.
- The `without_structured_representation` ablation uses the same generated observables with unrestricted candidates and the full bootstrap contract.
- All graph methods receive matched intervention evaluation; empty graphs produce missing metrics plus a reason.

## Runtime, provenance, and rendering

Automatic worker selection is capped at `min(cpu-1, 12)`. Per-method runtimes and computed speedups are stored separately from semantic generation time. The run manifest records semantic freeze, prediction freeze, baseline-simulation start, temporal completion, intervention-simulation start, and intervention completion.

The Figure 2--8 renderer migrates the local original visual language into a dynamic adapter. It uses the frozen graph-error opacity, mean intervention curves, full effect-matrix range, median repeated-subsample curves, and median nested 95% interval endpoints. No historical scientific value is hard-coded.

## Verification boundary

Unit tests, the toy smoke check, and the two-simulator development end-to-end check are non-scientific software validation. They call no real model API and write outside `runs/`. A formal experiment has not been executed by the implementation process.
