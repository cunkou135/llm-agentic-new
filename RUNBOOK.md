# 正式运行手册

## 1. 安装与 API 配置

在本目录打开 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test]"
Copy-Item config\llm_api.example.json config\llm_api.local.json
```

编辑 `config\llm_api.local.json`，填写 OpenAI-compatible endpoint、密钥、模型、
单次输出上限、timeout 和 retry。该本地文件已被忽略；provenance 只保存脱敏配置。
正式运行缺少密钥时会失败，不会自动换成 mock。

## 2. 正式运行

必须使用从未出现过的 run ID。旧 `res_f` 和旧 seed 仅作为 development
evidence，不得 resume、复制或拼接到新结果。

```powershell
.\.venv\Scripts\python run_experiment.py `
  --config config\experiment.json `
  --llm-config config\llm_api.local.json `
  --run-id confirmatory_path_v4_001 `
  --workers auto `
  --plot-repo "..\llm-agentic-dis"
```

正式阶段固定为：

```text
indicator_generation
indicator_freeze
path_generation
semantic_freeze
baseline_simulation
temporal
path_temporal_qualification
intervention_simulation
intervention
path_intervention_classification
prospective
primary_freeze
dose_response
holdout_simulation
holdout_confirmation
temporal_negative_control
robustness
export
render
```

前 6 次 LLM 调用只生成 indicators。前 3 次可参与 data-blind selection，后 3
次仅用于 replication。指标冻结后，第二类调用使用 1 个 primary 加 2 个
replication-only generations 生成完整路径；只有第一个 accepted primary 进入后续
实验。Phase B 同时生成 6 条绑定 `candidate_path_id` 的 prospective predictions，
并在 baseline 前完成哈希冻结。后面的 `prospective` stage 只是验证，不重新生成。

## 3. 分阶段恢复

通常应运行完整命令。中断后使用原 run ID 和完全相同的配置、代码、模型设置，
加 `--resume`。如需只恢复一个阶段：

```powershell
.\.venv\Scripts\python run_experiment.py `
  --config config\experiment.json `
  --llm-config config\llm_api.local.json `
  --run-id confirmatory_path_v4_001 `
  --workers auto `
  --stage path_generation `
  --resume
```

把 `path_generation` 换成需要恢复的阶段名即可。依赖阶段未完成会失败。每轮
LLM 的 prompt、request、response、accepted payload 和 SHA256 均保存在
`llm\indicator\...` 或 `llm\path\...`。任何不一致都会 fail closed；baseline
开始后禁止重跑语义阶段。

实时进度：

```text
runs\confirmatory_path_v4_001\logs\progress.jsonl
```

## 4. 科学判定口径

- Phase A 固定 16 Micro、8 Meso、4 Macro，不输出关系、路径或预测。
- Phase B 只能引用冻结 ID；每场景固定 16--24 条完整路径。
- derived candidate edges 只由 frozen paths 自动投影和去重。
- Stage 2 仍估计相邻关系，但 BH/FDR family 由路径的 Macro endpoint 事前定义。
  两条关系在对应 Macro group 内都 retained，完整路径才 temporally qualified。
- Stage 3 只改变真实 simulator parameter。只有 manipulation、三尺度响应、两条
  relation evidence、冻结方向、严格 onset order 和 observational lag 全部满足时，
  路径才是 `supported`。证据不足是 `inconclusive`，不得并入 supported。
- `baseline_sd=0` 的标准化效应为 NaN、significant 为 False、onset 为 -1；raw
  effect 保留。
- Primary 只使用 minus/plus；mid doses 只进入 secondary dose analysis。
- Holdout 只确认 frozen primary paths，不生成 replacement path，不修改 primary
  classification、lag 或 threshold。
- Representative 只能来自 frozen supported paths。优先 holdout-confirmed 后按
  path ID 排序；无候选时保存 null。
- Figure 5/7 只绘制真实 supported frozen path，不跨 parameter、direction、root
  或 hypothesis group 借用 evidence。

## 5. Seeds 与任务数

- Primary：3101--3124，共 24 个。
- Holdout：4101--4112，共 12 个，和 primary 不重叠。
- Primary：每场景 24 seeds x（baseline + 12 dose intervention conditions +
  mechanism-disabled）= 672 trajectories。
- Holdout：每场景 12 seeds x（baseline + 6 extreme intervention conditions +
  mechanism-disabled）= 192 trajectories。
- 总计：864 trajectories。

## 6. 结果位置

所有正式结果位于：

```text
runs\confirmatory_path_v4_001\
```

重点目录：

- `config\`：冻结的实验与脱敏 API 配置。
- `provenance\`：source、stage、artifact hashes 和环境记录。
- `llm\indicator\`、`llm\path\`：两个独立 LLM 阶段的完整历史。
- `representation\`：冻结 indicators、paths、derived edges、predictions 和
  replication metrics。
- `data\primary\raw_logs\`：公开 primary NPZ。
- `data\holdout\raw_logs\`：物理隔离的 holdout NPZ。
- `data\**\reference_hidden\`：仅 Controlled Recovery 可用的隐藏参考文件。
- `analysis\`：temporal、intervention、path funnel、dose、holdout、negative
  control、robustness、prospective 和 attribution 输出。
- `visualization_input\`、`figures\`、`tables\`：动态导出和论文图表。

不要手工修改 run 内 JSON、CSV、Parquet、NPZ 或哈希 sidecar。

## 7. 开发检查

```powershell
python -m pytest -q
python smoke_pipeline.py --run-id smoke_local --workers 1
python run_dev_e2e.py --run-id dev_local --workers 2
```

开发运行使用 mock LLM、较少 seeds 和较少 bootstrap，只验证执行合同。输出位于
`smoke_runs\` 或 `dev_runs\`，明确不是科学证据。不得因开发结果是否出现
supported path 而调整 FDR、support、effect、onset 或 lag tolerance。

如渲染阶段需要单独恢复：

```powershell
.\.venv\Scripts\python render_paper_figures.py `
  --run runs\confirmatory_path_v4_001 `
  --plot-repo "..\llm-agentic-dis" `
  --formats png svg pdf
```
