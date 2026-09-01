# Reproducible multiscale emergence attribution experiments

This is a new, self-contained scientific experiment implementation. It starts from simulator rules and raw logs, obtains executable multiscale semantic hypotheses and prospective predictions from one OpenAI-compatible API boundary, qualifies proposed temporal relations over repeated trajectories, tests propagation with matched-seed parameter interventions, and exports paper-data artifacts and Figure 2--8 inputs.

The project keeps three evidence types separate:

- semantic hypotheses proposed before quantitative analysis;
- temporal evidence from repeated baseline trajectories;
- intervention evidence from actual simulator parameter changes.

A temporally qualified relation is not labelled as an unrestricted causal relation.

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
  --no-render
```

Resume an interrupted run by adding `--resume`. Generate Figure 2--8 afterward with `render_paper_figures.py`; see `RUNBOOK.md` for the exact commands and artifact map.

## Integrity gates

- Formal semantic generation fails closed if the API key is absent.
- The prompt receives no baseline numerical summary or evaluation information.
- Model output is declarative JSON AST only; no generated code is evaluated.
- All three independent generations, repairs, validation errors, and selection reasons are preserved.
- Parameter associations are model-proposed and never inserted by a deterministic supervisor.
- Reference structures are available only to simulator internals and final evaluation.
- The prospective prediction file is hashed before downstream analysis.
- Source, configuration, prompt, API settings without the key, stage outputs, and final artifacts are hashed.
- An existing run identifier is never silently overwritten.
- Toy smoke outputs are physically separate and explicitly marked non-scientific.

## Software checks

```powershell
python -m pytest -q
python smoke_pipeline.py --run-id smoke_local --workers 1
```

The first command runs unit tests. The second runs only the deterministic toy wiring path and must never be used as scientific evidence.

