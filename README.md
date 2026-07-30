# SWE-Pruner Training-Free Alternatives

这是一个面向 coding agent 的训练免费上下文剪枝实验仓库。它把研究方案中的六条基础路线拆成独立任务，另提供 IR+AST 融合组，并包含真实 mini-swe-agent + SWE-Bench Verified 端到端入口。

“Training-free”在这里指：**不新增训练好的专用 observation model 或 pruning head**。PPL、hidden state、attention 和 influence 方法仍可读取冻结 backbone 的推理信号，但绝不会更新模型参数。

真实 coding-agent 主实验隔离在 `zero_forward_pruning/`。它在工具输出
进入 Qwen 之前，使用现有任务、工具命令、identifier、错误信号和代码
结构生成可恢复的紧凑视图。该子系统不导入 vLLM 客户端，不调用大模型，
每次剪枝的 model forward 和 LLM token 都严格为 0。完整设计见
`docs/ZERO_FORWARD_PRUNING_DESIGN.md`。

## 实现与端到端实验组

| 方法 | 独立目录 | 训练 | 运行时模型/内部信号 | 主要用途 |
|---|---|---:|---|---|
| IR + 结构锚点 | `tasks/ir_structural` | 无 | 无 | closed/open agent 的首选低成本基线 |
| 条件 PPL + surprisal | `tasks/conditional_ppl` | 无 | 本地 causal LM | 模型感知的外部 scorer |
| Hidden-state 锚点相似度 | `tasks/hidden_state_similarity` | 无 | hidden states | 最接近 SWE-Pruner Pro 的 head 替代 |
| Attention rollout / heavy hitter | `tasks/attention_rollout` | 无 | attention tensors | open-weight 推理引擎实验 |
| 影响函数式贪心删除 | `tasks/influence_oracle` | 无 | 多次冻结模型 forward | 离线 oracle 与排序上界 |
| 执行信号 + AST 骨架 | `tasks/execution_ast` | 无 | 可选 parser | 稳健回退与 rank fusion |
| IR + AST rank fusion | `tasks/ir_ast_hybrid` | 无 | 无 | 真实 agent 主实验融合组 |

实现优先级与研究报告一致：IR、hidden-state、influence 是前三组主实验；另外三条也保留为可运行、可评测的独立实现。

## 快速运行

要求 Python 3.11+。

```bash
python3 -m pip install -e '.[dev]'

python3 -m tf_pruning.cli methods

python3 -m tf_pruning.cli prune \
  --method ir_structural \
  --request examples/requests/source_file.json

python3 -m tf_pruning.cli evaluate \
  --method execution_ast \
  --input examples/replay/demo.jsonl \
  --output-dir outputs/execution_ast_demo \
  --budget-schedule configs/length_aware_budget.json
```

服务器环境安装完成、vLLM 已在 `8015` 端口提供
`Qwen3.5-27B`、mini-swe-agent 已安装后，真实 coding-agent 主实验只使用
零额外模型调用入口：

```bash
git pull --ff-only
bash scripts/create_server_conda.sh
cp configs/zero_forward_server_profile.example.env zero_forward_server_profile.env
# 编辑 MINI_SWE_PYTHON；自定义 mini YAML 时再设置 MINI_SWE_BASE_CONFIG
bash scripts/run_zero_forward_swebench.sh preflight
bash scripts/run_zero_forward_swebench.sh smoke
```

环境创建脚本与启动器都会先移除当前终端继承的 uv/venv，再激活
`swepruner-training-free` conda 环境。一题 smoke 在相同 SWE-Bench
Verified 任务上比较 baseline 与 `adaptive_evidence`；它不回退到 toy
replay，也不向 Qwen 发送任何额外的剪枝请求。查看状态、汇总和官方评分：

```bash
bash scripts/run_zero_forward_swebench.sh status
bash scripts/run_zero_forward_swebench.sh results
bash scripts/run_zero_forward_swebench.sh grade
```

smoke 通过后运行 `launch`，默认生成 baseline、safe rules、intent IR、
intent structure、adaptive evidence 五组实验。只有 agent 的正常推理
连接 `:8015`；四个 pruner 都是纯 CPU。model id 会从
`http://127.0.0.1:8015/v1/models` 自动发现。服务器本地 agent 应只按
`docs/ZERO_FORWARD_SERVER_HANDOFF.md` 的逐项交接与验收清单操作。

`scripts/run_server_experiments.sh` 和 `tasks/` 下的旧路线保留用于离线
ablation/replay 复现，不是这次零 forward 在线主实验的入口。尤其是需要
冻结模型多次 forward 的 oracle 类方法，不会被
`run_zero_forward_swebench.sh` 启动。

兼容官方 `POST /prune` 请求/响应字段的本地服务：

```bash
tf-prune-serve \
  --method ir_structural \
  --config tasks/ir_structural/config.example.json \
  --host 127.0.0.1 \
  --port 8000
```

原有 tool wrapper 继续发送 `{"query": "...", "code": "..."}` 即可。
训练免费方法使用 hard budget，没有校准后的模型 probability；兼容服务会
把官方 `threshold=t` 单调映射为 `keep_ratio=1-t`。正式实验推荐直接
发送扩展字段 `keep_ratio`，避免混淆两种语义。服务默认 fail-open，
pruner 缺信号或异常时返回原 observation 并填写 `error_msg`。

需要冻结模型或数组处理的任务：

```bash
python3 -m pip install -e '.[model,dev]'
```

从服务器本地冻结模型生成 hidden-state 与 decode-attention 输入：

```bash
python scripts/extract_hf_signals.py \
  --model-path /local/models/CODER_MODEL \
  --request examples/requests/source_file.json \
  --output outputs/source_file_signals.npz \
  --signals both
```

命令还会生成可直接交给统一 CLI 的 companion request JSON。模型加载
固定为 local-only，参数冻结，长输入超限会显式失败而不会静默截断。
完成 editable install 后也可使用等价入口 `tf-extract-signals`。

需要可选 tree-sitter 多语言解析：

```bash
python3 -m pip install -e '.[syntax,dev]'
```

模型任务默认只接受本地模型目录或本地 `.npz`，不会隐式访问 Hugging Face Hub。

## 目录结构

```text
training_free_pruning/
├── START_HERE.md
├── README.md
├── pyproject.toml
├── tf_pruning/                 # 共享协议、预算、选择、注册与 CLI
├── tasks/
│   ├── ir_structural/
│   ├── conditional_ppl/
│   ├── hidden_state_similarity/
│   ├── attention_rollout/
│   ├── influence_oracle/
│   ├── execution_ast/
│   └── ir_ast_hybrid/
├── agent_eval/                 # mini-swe-agent 配置适配、轨迹/评分汇总
├── evaluation/                 # replay 与代理指标
├── integrations/               # fail-open middleware 与官方 HTTP 兼容层
├── zero_forward_pruning/       # 隔离的零模型调用协议、方法、恢复服务与 mini adapter
├── configs/                    # 长度预算与 Pareto 预算
├── examples/                   # 请求和 replay 样例
├── scripts/                    # 本地/服务器/实验脚本
├── docs/                       # 输入、实验、集成、服务器说明
└── tests/
```

每个方法都实现统一的 `Pruner.prune(PruningRequest) -> PruningResult`。结果包含：

- 原始行数与保留行号；
- 带显式省略标记的骨架文本；
- 逐行分数和可审计的保留原因；
- 单次耗时、模型 forward 次数及方法元数据。

## 离线 replay

Replay JSONL 每行包含 `request`、可选 `gold_line_numbers` 和必须保护的 `required_line_numbers`。统一评测会生成：

```text
OUTPUT_DIR/
├── results.jsonl
├── per_sample_metrics.jsonl
├── errors.jsonl
└── summary.json
```

核心代理指标包括 line precision/recall/F1、required-line recall、critical miss rate、行与估算 token 保留率、方法耗时和模型 forward 次数。真实 agent 脚本会从 trajectory 汇总端到端 token、API calls、观测剪枝率和错误数；运行官方 grader 后填入 Resolve Rate。未评分时 Resolve Rate 明确显示为空，不会把 `Submitted` 或离线 recall 冒充任务成功率。

现有父项目的 `pruning_sft.jsonl`、`swe_pruner_compatible.jsonl` 和
official-format JSONL 可用 `python -m evaluation.convert_existing`
直接转换，详见 `docs/EXPERIMENT_GUIDE.md`。

按研究报告的五档预算跑 Pareto：

```bash
bash scripts/run_replay_matrix.sh \
  ir_structural \
  examples/replay/demo.jsonl \
  outputs/ir_matrix
```

该脚本还会汇总 `matrix.json` 与 `matrix.csv`，并按 required-line
recall（越高越好）和估算 token retention（越低越好）标记 Pareto 点。

## 与现有项目/服务器一致

服务器继续使用现有约定：

- Python 3.11；
- B200；
- PyTorch CUDA 13.0 wheel；
- 默认服务器根目录 `/home/yuantao/futao`；
- 模型和数据由服务器下载或放入本地目录；
- GPU 通过 `CUDA_VISIBLE_DEVICES` 显式选择。

安装和检查命令见 `docs/SERVER_GUIDE.md`。本目录是独立仓库边界，不包含
父项目的训练数据、checkpoint、历史 artifacts 或 Git 记录。发布门禁见
`docs/PUBLISH_CHECKLIST.md`。

## 验证

研究方案每一项对应到哪些类、测试和实验脚本，见
`docs/IMPLEMENTATION_MAP.md`。

```bash
bash scripts/run_local_checks.sh
```

该命令执行 Python 编译、全部单元测试与统一 CLI 方法发现。模型大权重不属于单元测试前置条件；各模型方法用 deterministic mock 验证排序与预算逻辑。GitHub Actions 还会在 Python 3.11/3.12/3.13 上运行 Ruff、完整测试、Shell 和 console-entrypoint 检查。
