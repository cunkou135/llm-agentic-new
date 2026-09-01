# Reproducible multiscale emergence attribution experiments

This is the existing, self-contained scientific experiment implementation. It starts from simulator rules and public raw logs, freezes executable multiscale semantic hypotheses and prospective predictions before simulation, qualifies proposed temporal relations over repeated trajectories, tests propagation with matched-seed parameter interventions, and exports paper-data artifacts and Figure 2--8 inputs.

The project keeps three evidence types separate:

- semantic hypotheses proposed before quantitative analysis;
- temporal evidence from repeated baseline trajectories;
- intervention evidence from actual simulator parameter changes.

A temporally qualified relation is not labelled as an unrestricted causal relation.

Stage 3 changes simulator parameters only.  Direct Micro->Meso evidence uses
`parameter -> Micro -> Meso`; Meso->Macro evidence reuses the same upstream
Micro manipulation root and is labelled `upstream_mediated`.  Generated Micro,
Meso, and Macro indicators are observed from the matched-seed trajectories and
are never directly edited.

## Formal run

Create an environment and install the project:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test]"
Copy-Item config\llm_api.example.json config\llm_api.local.json
```

Fill `config/llm_api.local.json`, then run:

```powershell
python run_experiment.py `
  --config config\experiment.json `
  --llm-config config\llm_api.local.json `
  --run-id rerun_001 `
  --workers auto `
  --plot-repo "..\llm-agentic-dis"
```

Resume an interrupted run by adding `--resume`. Accepted semantic generations
are skipped only after their request, response, accepted payload, and result
hashes verify; existing LLM history is append-only. Generate Figure 2--8
afterward with `render_paper_figures.py`; see `RUNBOOK.md` for the exact commands
and artifact map.

## Integrity gates

- Formal semantic generation fails closed if the API key is absent.
- The prompt receives no baseline numerical summary or evaluation information.
- Model output is declarative JSON AST only; no generated code is evaluated.
- All three independent generations, repairs, validation errors, and selection reasons are preserved.
- Parameter associations are model-proposed and never inserted by a deterministic supervisor.
- Withheld reference states are physically separated under `data/reference_hidden/` and are used only by the labelled Controlled Recovery track.
- Full Discovery never aligns generated nodes to withheld reference identities and therefore reports no reference F1, SHD, lag MAE, or direction score.
- The prospective prediction file is hashed before downstream analysis.
- Missing required prospective responses, wrong directions, temporal-order
  violations, and missing required observational edges are falsifications.
- `analysis/attribution_objects.json` integrates `full_method` evidence only;
  other methods are exported separately for comparison.
- Unrestricted temporal search removes both structured candidate-edge
  constraints and semantic branch constraints. It preserves the same lag
  range, OLS formulation, screening threshold, BH procedure, whole-trajectory
  bootstrap repetitions, and support threshold; the hypothesis space and FDR
  grouping are intentionally no longer structure-constrained.
- Intervention attempts are frozen to an edge-level rule: any explicit
  directional contradiction overrides support; otherwise at least one support
  is sufficient even if another applicable attempt is a manipulation failure.
- A mechanism-disabled check targets one named mechanism pathway. Its overall
  retained-graph statistics do not imply that all unrelated dynamics should
  disappear.
- Source, configuration, prompt, API settings without the key, stage outputs, and final artifacts are hashed.
- An existing run identifier is never silently overwritten.
- Toy smoke outputs are physically separate and explicitly marked non-scientific.

## Software checks

```powershell
python -m pytest -q
python smoke_pipeline.py --run-id smoke_local --workers 1
python benchmark_robustness_pool.py --run-id pool_smoke_local --workers 2
```

The first command runs unit tests. The other commands are deterministic,
NON_SCIENTIFIC wiring and process-lifecycle checks and must never be used as
scientific evidence.
