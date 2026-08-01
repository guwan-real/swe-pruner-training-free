# Posterior History：服务器交接与运行

此方案独立于 `zero_forward_pruning/`：不启动 HTTP pruner，不占用 8111–8124，也不改动已经存在的 SWE-Pruner 服务。它只使用现有：

- Qwen3.5-27B vLLM：`http://127.0.0.1:8015/v1`；
- SWE-Pruner 官方仓库中的 `mini-swe-agent--with-pruning`；
- Docker 和 SWE-Bench harness。

## 配置

从 `configs/posterior_history_server_profile.example.env` 创建服务器本地的 `posterior_history_server_profile.env`。它被 Git 忽略；填写现有 mini 的 Python 路径，必要时填写 `MINI_SWE_BASE_CONFIG` 与 `SWEBENCH_PYTHON`。

运行入口是 `scripts/run_posterior_history_swebench.sh`。它在激活 conda 前主动移除继承的 uv/venv，并使用 `MINI_SWE_PYTHON` 调用已有的 official fork。默认 `PARALLEL_ARMS=0`，因为并发 arm 会争用同一 vLLM，无法公平比较 wall time。

launcher 固定 `AGENT_STEP_LIMIT=100`，并覆盖 base YAML 中缺失、为 `0`（无限）或
其他值的 `agent.step_limit`。preflight 会拒绝非 100 的实验配置，最终值同时写入
`configs/agent.yaml`、`configs/agent.yaml.meta.json` 和 `manifest.json`。

## Arms

默认只包含：

```text
baseline
posterior_adaptive
```

两者通过同一个生成的 mini YAML 启动；二者都保留 fork 原有的 `context_focus_question` prompt，但都移除 legacy `agent.pruner`。差异仅在于 `posterior_adaptive` 启用历史 view hook。

可将 `POSTERIOR_HISTORY_METHODS=safe,adaptive` 加入保守 selector 消融。不要将旧的 zero-forward arms 与本实验混到同一 run root。

## 命令模式

- `preflight`：检查 conda、Docker、vLLM、official fork 的 `query -> model.query(messages)` 契约和 base YAML；
- `smoke`：1 个真实 SWE-Bench task，顺序运行 baseline 与 posterior_adaptive；
- `launch`：按 profile 的 `TASK_SLICE` 启动；
- `status`：读取 pid 和 arm 状态；
- `results`：写出 `summary.csv/json` 与 baseline 差值；
- `grade`：对每个有 `preds.json` 的 arm 做官方 harness 评分；
- `stop`：停止该 run 的 arm 进程。

## 复用旧 baseline：1000 / 500 串行实验

已有同模型、同 prompt、同任务切片的 baseline 时，不需要再次花费时间运行它。
一键脚本默认只运行 `posterior_adaptive`，先 1000、完成后再 500：

```bash
bash scripts/run_posterior_threshold_sweep.sh launch
```

它固定 `TASK_SLICE=0:20`、`PARALLEL_ARMS=0`、`SKIP_BASELINE=1` 和
`POSTERIOR_HISTORY_METHODS=adaptive`。两个 run tag 会写入仓库本地且 Git 忽略的
`.posterior_threshold_sweep.last`。之后可直接批量查看结果或评分：

```bash
bash scripts/run_posterior_threshold_sweep.sh results
GRADER_WORKERS=32 bash scripts/run_posterior_threshold_sweep.sh grade
```

需要只跑一个阈值时，在 `launch` 后传值，例如
`bash scripts/run_posterior_threshold_sweep.sh launch 1000`。可用
`RUN_TAG_PREFIX` 自定义两轮共同前缀。

launcher 的配置优先级为“显式命令环境 > server profile > 内置默认值”。因此即使
本地 profile 写着 `POSTERIOR_MIN_INPUT_TOKENS=1500`，sweep 传入的 1000/500
也不会被覆盖。每个 run 的 `manifest.json` 会记录最终生效的阈值、
`baseline_included=false` 和每题 100 次模型调用上限。posterior-only 的
`results` 不要求 run 目录中存在 baseline；相对 baseline 的 delta 字段留空，
后续与旧 baseline 统一比较即可。

## Smoke 通过条件

先看 trajectory：

1. 第一条工具 observation 在紧随其后的 model request 中必须保持全文；
2. 只有存在至少 `POSTERIOR_HOT_OBSERVATIONS + 1` 条 observation 后，最早的记录才出现 `posterior_history_compaction`；
3. trajectory 中的原始 `Observation:` 内容仍是全文；
4. `summary.csv` 中 `pruner_model_forwards=0`、`pruner_llm_tokens=0`；
5. baseline 与 posterior 的 prompt config / CFQ 使用次数相同。

`summary.csv` 还会分别报告 `history_observations_seen`、
`history_observations_tracked` 和 `history_observations_untracked`。官方
`<output_head>/<output_tail>` 长输出在 trajectory 中应显示
`boundary_mode=official-head-tail`，不能再静默消失；
`history_observations_untracked>0` 时先检查 trajectory 中的模板格式，不要通过
降低阈值掩盖边界问题。

调用次数要看 `agent_api_calls_mean_per_task` 和 `agent_api_calls_max_per_task`；
`agent_api_calls` 是整个 task slice 的合计。`agent_step_limit_hits` 或
`agent_step_limit_exits` 大于 0 表示确实有任务运行到了 100 次硬上限。

本版本的 `history_token_estimators` 应为
`max-lexical-ascii4-unicode1-v2`。若仍然没有压缩，先比较
`posterior_below_min_input_observations` 和
`posterior_no_safe_reduction_observations`，再查看对应 message 的
`hard_block_count / matched_block_count / selected_block_count`。不要先把
`POSTERIOR_MIN_INPUT_TOKENS` 从 1500 一步降到 500。

然后才比较 5 题和 20 题：主接受标准是 API calls 不增加、prompt tokens 降低，并维持官方 resolve rate。若 agent calls 上升，先用同一轨迹中的 `posterior_history_stats` 定位是哪一个冷历史 observation 造成重读，不要先降低保留阈值。
