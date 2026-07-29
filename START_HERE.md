# 从这里开始

这个仓库是一套 training-free coding-agent 上下文剪枝实验。它不会训练
观察模型、分类头或 pruning head。基础方法共享统一接口；服务器主入口
实际运行 mini-swe-agent + SWE-Bench，不再把两行 demo 当成 agent 实验。

## 第一步：先跑纯 CPU 方法

```bash
cd training_free_pruning
python3 -m pip install -e '.[dev]'
python3 -m tf_pruning.cli methods
python3 -m tf_pruning.cli prune \
  --method ir_structural \
  --request examples/requests/source_file.json
python3 -m tf_pruning.cli prune \
  --method hidden_state_similarity \
  --config tasks/hidden_state_similarity/config.example.json \
  --request examples/requests/hidden_state.json
python3 -m tf_pruning.cli prune \
  --method attention_rollout \
  --config tasks/attention_rollout/config.example.json \
  --request examples/requests/attention.json
python3 -m tf_pruning.cli evaluate \
  --method execution_ast \
  --input examples/replay/demo.jsonl \
  --output-dir outputs/ast_demo \
  --keep-ratio 0.5 \
  --no-prune-below 0
```

IR 与执行信号/AST 两个任务不需要 GPU。其余任务需要预计算的 hidden-state/attention 数组，或显式配置本地模型路径；所有模型加载都必须是 local-only。

接现有 SWE-Pruner tool wrapper 时，可直接启动兼容 endpoint：

```bash
tf-prune-serve \
  --method ir_structural \
  --config tasks/ir_structural/config.example.json
```

默认地址就是官方示例使用的 `http://127.0.0.1:8000/prune`。

## 第二步：选择任务

| 目录 | 适用条件 | 首次建议 |
|---|---|---|
| `tasks/ir_structural` | 任意 closed/open agent | 主实验 1 |
| `tasks/hidden_state_similarity` | 能取得 backbone hidden states | 主实验 2 |
| `tasks/influence_oracle` | 能运行本地 backbone，离线小样本 | 主实验 3 / oracle |
| `tasks/conditional_ppl` | 能运行本地 causal LM | 第二阶段 |
| `tasks/attention_rollout` | 推理引擎能导出 attention | 第二阶段 |
| `tasks/execution_ast` | 任意 agent，尤其日志/grep/diff | 通用回退 |
| `tasks/ir_ast_hybrid` | 任意 agent | IR 与执行证据融合主实验 |

每个任务目录内都有独立 `README.md` 与 `config.example.json`。

## 第三步：本地验证

```bash
bash scripts/run_local_checks.sh
```

服务器已经启动 `qwen3.5-27b` vLLM（端口 `8015`）且安装了带 pruning
hook 的 mini-swe-agent 时，先做真实一题 smoke：

```bash
bash scripts/run_server_experiments.sh preflight
bash scripts/run_server_experiments.sh smoke
```

它会自动关闭继承的 uv/venv、激活 conda、读取 vLLM model id，运行
baseline/IR 两组相同的单个 SWE-Bench Verified 任务。随后使用同一脚本
的 `status`、`results`、`grade` 子命令查看轨迹统计与官方 Resolve
Rate。通过后再用零参数入口运行默认前 10 题的
baseline/IR/AST/IR+AST；两者都不是 replay。

完整数据格式见 `docs/INPUT_FORMAT.md`，逐项实现证据见
`docs/IMPLEMENTATION_MAP.md`，实验步骤见 `docs/EXPERIMENT_GUIDE.md`，
服务器部署见 `docs/SERVER_GUIDE.md`，接入 agent 的位置见
`docs/INTEGRATION_GUIDE.md`，实际多实验参数见
`docs/CODING_AGENT_EXPERIMENTS.md`。让服务器本地 agent 接手时，直接
给它阅读 `docs/SERVER_AGENT_HANDOFF.md`。
