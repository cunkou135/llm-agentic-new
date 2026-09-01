# 正式运行手册

## 0. 一次性安装

在本目录打开 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test]"
Copy-Item config\llm_api.example.json config\llm_api.local.json
```

## 第一步：填写 API

编辑 `config\llm_api.local.json`：

```json
{
  "base_url": "你的 OpenAI-compatible base URL",
  "api_key": "你的密钥",
  "model": "你的模型名",
  "temperature": 0.1,
  "max_tokens": 12000,
  "timeout": 300,
  "max_retries": 2
}
```

只有 `src\emergence_attribution\llm_client.py` 会访问模型 API。local 文件已加入 `.gitignore`；provenance 只保存脱敏配置，不保存明文密钥。缺少密钥时 semantic stage 会直接失败，不会用 mock 替代正式输出。

## 第二步：执行正式实验（唯一推荐命令）

该命令按冻结顺序一次完成语义、两段仿真、分析、导出与 Figure 2--8 渲染：

```powershell
.\.venv\Scripts\python run_experiment.py `
  --config config\experiment.json `
  --llm-config config\llm_api.local.json `
  --run-id rerun_001 `
  --workers auto `
  --plot-repo "..\llm-agentic-dis"
```

正式阶段顺序固定为：

```text
semantic -> baseline_simulation -> temporal -> intervention_simulation -> intervention -> prospective -> robustness -> export -> render
```

如需显式指定 12 个进程，可把 `--workers auto` 改为 `--workers 12`。`auto` 会采用 `min(cpu-1, 12)` 的内存友好上限。

用于故障恢复的分阶段命令如下；正常首次运行不要使用这些命令：

```powershell
.\.venv\Scripts\python run_experiment.py --config config\experiment.json --llm-config config\llm_api.local.json --run-id rerun_001 --workers auto --stage semantic
.\.venv\Scripts\python run_experiment.py --config config\experiment.json --llm-config config\llm_api.local.json --run-id rerun_001 --workers auto --stage baseline_simulation --resume
.\.venv\Scripts\python run_experiment.py --config config\experiment.json --llm-config config\llm_api.local.json --run-id rerun_001 --workers auto --stage temporal --resume
.\.venv\Scripts\python run_experiment.py --config config\experiment.json --llm-config config\llm_api.local.json --run-id rerun_001 --workers auto --stage intervention_simulation --resume
.\.venv\Scripts\python run_experiment.py --config config\experiment.json --llm-config config\llm_api.local.json --run-id rerun_001 --workers auto --stage intervention --resume
.\.venv\Scripts\python run_experiment.py --config config\experiment.json --llm-config config\llm_api.local.json --run-id rerun_001 --workers auto --stage prospective --resume
.\.venv\Scripts\python run_experiment.py --config config\experiment.json --llm-config config\llm_api.local.json --run-id rerun_001 --workers auto --stage robustness --resume
.\.venv\Scripts\python run_experiment.py --config config\experiment.json --llm-config config\llm_api.local.json --run-id rerun_001 --workers auto --stage export --resume
```

## 第三步：查看实时进度

终端会显示 Overall experiment、当前 stage、completed/total、百分比、elapsed、ETA、workers 和当前 scenario/parameter/replicate。

持久进度日志位于：

```text
runs\rerun_001\logs\progress.jsonl
```

即使进程异常退出，该 JSONL 也保留最后完成的位置。

## 第四步：失败后 resume

使用原来的 config、LLM model settings、代码和 run-id，加 `--resume`：

```powershell
.\.venv\Scripts\python run_experiment.py `
  --config config\experiment.json `
  --llm-config config\llm_api.local.json `
  --run-id rerun_001 `
  --workers auto `
  --resume `
  --plot-repo "..\llm-agentic-dis"
```

Semantic resume 是 generation-level：若某个
`llm\<scenario>\generation_*\generation_result.json` 已接受，程序会先校验
prompt/request/response/accepted payload/result 的哈希，然后跳过该 generation。
缺失 sidecar、哈希不一致或当前 prompt contract 不一致都会 fail closed；已有
request/response 不会被静默覆盖。

## Stage 3 与指标口径

- Micro->Meso 使用 direct Micro manipulation root，范围标记为 `direct_root`。
- Meso->Macro 沿同一候选/保留路径寻找上游 Micro root，仅修改其真实 simulator parameter，范围标记为 `upstream_mediated`。
- `directionally_contradicted` 是唯一的方向矛盾名称。
- Full Discovery 的 contradiction rate 分母为除 `not_applicable` 外的全部适用尝试；`manipulation_failure` 保留在分母中。
- Controlled intervention recall 的分母仅包含具有合法 manipulation route 的 truth edges；其余 truth edges 为 `not_applicable`，不是 false negative。
- unrestricted temporal qualification 使用实际 `N*(N-1)` candidate space；28 个节点时分母为 756。
- unrestricted temporal search 同时移除 structured candidate-edge constraints
  和 semantic branch constraints；lag range、OLS、screening threshold、BH
  procedure、whole-trajectory bootstrap repetitions 与 support threshold 保持
  不变。也就是说核心估计/重采样过程一致，但 hypothesis space 与 semantic
  FDR grouping 不再受结构约束。
- `without_structured_representation` 使用同一定义，不应描述为“仅 candidate
  space 不同”。
- 干预先按 source->target 聚合：任何 `directionally_contradicted` 优先于
  `supported`；若无明确矛盾，至少一个 `supported` 可覆盖另一侧的
  `manipulation_failure`。
- `mechanism_disabled_checks.csv` 的整体 retained-graph 指标只描述一个目标
  机制被禁用后的点图；不要求或暗示所有无关系统 dynamics 消失。
- Figure 7 只接受 Micro/Meso/Macro 均显著、onset 有序且两条 Full Method intervention edge 均为 `supported` 的路径。

程序会核对 source/config/API-public-config/stage artifact hashes。已经验证成功的 task 或 stage 会跳过；hash 不一致会 fail closed，并要求新 run-id。不要手工修改 run 内 CSV、JSON、Parquet、NPZ 或图数据。

## 第五步：正式数据保存位置

全部正式数据位于：

```text
runs\rerun_001\
```

关键目录：

- `config\`：冻结的 experiment snapshot 和脱敏 API 配置。
- `provenance\`：run/source/stage manifests、环境、全部 artifact SHA256。
- `llm\`：两个场景各三代的 prompt、request、response、repair 和 validation。
- `representation\`：最终表征、agreement、validation、冻结 prospective predictions。
- `data\raw_logs\`：384 个仅含公开字段的正式原始仿真文件。
- `data\reference_hidden\`：与公开 NPZ 物理隔离的 Controlled Recovery 隐藏参考文件；不得提供给语义生成或 Full Discovery。
- `data\indicator_trajectories_*.parquet`：由冻结表征计算的 baseline/complete 指标轨迹。
- `analysis\full_discovery_results.csv`：不含隐藏真值对齐指标的发现轨结果。
- `analysis\controlled_recovery_results.csv`：固定隐藏已知基准上的恢复指标。
- `analysis\main_results.csv`：带 `evaluation_track` 标签的合并索引表。
- `analysis\`：其余 temporal、bootstrap、paired effects、intervention、robustness、prospective validation 和 attribution objects。
- `visualization_input\`：Figure 2--8 的动态输入 bundle。
- `figures\`、`tables\`：论文图和表格 source data。

## 第六步：生成 Figure 2--8

若正式命令因 `--no-render` 或渲染故障停在绘图前，可从冻结输入动态补绘：

```powershell
.\.venv\Scripts\python render_paper_figures.py `
  --run runs\rerun_001 `
  --plot-repo "..\llm-agentic-dis" `
  --formats png svg pdf
```

需要 TIFF 时加 `tiff`。渲染器动态读取实际 node/branch 名称，不依赖 frozen path、旧 node ID、旧 row count 或旧 hash。它明确使用 mean effect curve、完整 effect-matrix 色域，并以高可见度显示 added edges。

若要把数据 bundle 复制到本地绘图库供单独归档：

```powershell
.\.venv\Scripts\python export_visualization_bundle.py `
  --run runs\rerun_001 `
  --plot-repo "..\llm-agentic-dis"
```

该命令写入绘图库的 `data\generated_runs\rerun_001\`；若目标已存在会拒绝覆盖。科学绘图参数只读取 run 内冻结配置；本地绘图库仅作为原始视觉语言的已校验参考。

## 第七步：可直接用于论文的文件

优先使用：

- `analysis\main_results.csv`
- `analysis\full_discovery_results.csv`
- `analysis\controlled_recovery_results.csv`
- `analysis\main_graphs.jsonl`
- `analysis\data_efficiency_repeated_subsampling.csv`
- `analysis\paired_effects.parquet`
- `analysis\effect_curves.parquet`
- `analysis\intervention_classifications.csv`
- `analysis\comparative_method_intervention_evidence.csv`
- `analysis\path_timing_summary.csv`
- `analysis\observation_robustness.csv`
- `analysis\causal_scalability.csv`
- `analysis\prospective_validation.csv`
- `analysis\attribution_objects.json`
- `tables\*_source.csv`
- `figures\figure_2_*` 至 `figures\figure_8_*`
- `visualization_input\figure_inputs.generated.json`
- `visualization_input\SHA256SUMS`
- `figures\render_manifest.json`

只有 run 完成并生成 `RUN_FROZEN` 后，才把它作为完整正式 release。`smoke_runs\` 下的任何文件都不能用于论文。
