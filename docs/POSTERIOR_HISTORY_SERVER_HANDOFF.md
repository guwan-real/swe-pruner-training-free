# Posterior History：服务器交接与运行

此方案独立于 `zero_forward_pruning/`：不启动 HTTP pruner，不占用 8111–8124，也不改动已经存在的 SWE-Pruner 服务。它只使用现有：

- Qwen3.5-27B vLLM：`http://127.0.0.1:8015/v1`；
- SWE-Pruner 官方仓库中的 `mini-swe-agent--with-pruning`；
- Docker 和 SWE-Bench harness。

## 配置

从 `configs/posterior_history_server_profile.example.env` 创建服务器本地的 `posterior_history_server_profile.env`。它被 Git 忽略；填写现有 mini 的 Python 路径，必要时填写 `MINI_SWE_BASE_CONFIG` 与 `SWEBENCH_PYTHON`。

运行入口是 `scripts/run_posterior_history_swebench.sh`。它在激活 conda 前主动移除继承的 uv/venv，并使用 `MINI_SWE_PYTHON` 调用已有的 official fork。默认 `PARALLEL_ARMS=0`，因为并发 arm 会争用同一 vLLM，无法公平比较 wall time。

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

## Smoke 通过条件

先看 trajectory：

1. 第一条工具 observation 在紧随其后的 model request 中必须保持全文；
2. 只有存在至少 `POSTERIOR_HOT_OBSERVATIONS + 1` 条 observation 后，最早的记录才出现 `posterior_history_compaction`；
3. trajectory 中的原始 `Observation:` 内容仍是全文；
4. `summary.csv` 中 `pruner_model_forwards=0`、`pruner_llm_tokens=0`；
5. baseline 与 posterior 的 prompt config / CFQ 使用次数相同。

然后才比较 5 题和 20 题：主接受标准是 API calls 不增加、prompt tokens 降低，并维持官方 resolve rate。若 agent calls 上升，先用同一轨迹中的 `posterior_history_stats` 定位是哪一个冷历史 observation 造成重读，不要先降低保留阈值。
