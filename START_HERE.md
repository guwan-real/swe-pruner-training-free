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

服务器已经启动 `Qwen3.5-27B` vLLM（端口 `8015`）且已有
mini-swe-agent 时，零额外模型调用的工具边界入口就是主实验：

```bash
bash scripts/create_server_conda.sh
cp configs/zero_forward_server_profile.example.env zero_forward_server_profile.env
# 编辑 MINI_SWE_PYTHON；自定义 mini YAML 时再设置 MINI_SWE_BASE_CONFIG
bash scripts/run_zero_forward_swebench.sh preflight
bash scripts/run_zero_forward_swebench.sh smoke
bash scripts/run_zero_forward_swebench.sh launch
```

环境创建脚本和启动器会先关闭继承的 uv/venv，再激活项目 conda。该入口
在 `DefaultAgent.execute_actions()` 中拦截原始工具输出，在 observation
入模之前压缩；pruner forward/token 固定为 0，原文通过随机 ID 可恢复。
`smoke` 比较同一题的 baseline 与推荐方法，`launch` 运行五组真实
SWE-Bench 实验。时序、实验组与服务器配置见
`docs/ZERO_FORWARD_PRUNING_DESIGN.md` 和
`docs/ZERO_FORWARD_SERVER_HANDOFF.md`。

旧 `run_server_experiments.sh` 仅用于此前的 IR/AST/replay ablation，不
用于这次主实验；需要多次冻结模型 forward 的方法也不会由零 forward
启动器加载。

完整数据格式见 `docs/INPUT_FORMAT.md`，逐项实现证据见
`docs/IMPLEMENTATION_MAP.md`，实验步骤见 `docs/EXPERIMENT_GUIDE.md`，
服务器部署见 `docs/SERVER_GUIDE.md`，接入 agent 的位置见
`docs/INTEGRATION_GUIDE.md`。让服务器本地 agent 接手本次实验时，直接
给它阅读 `docs/ZERO_FORWARD_SERVER_HANDOFF.md`。
