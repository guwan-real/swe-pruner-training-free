# Coding-agent training-free 剪枝实验

## 默认比较组

| arm | Agent | observation 处理 | 训练 |
|---|---|---|---:|
| `baseline` | mini-swe-agent + Qwen3.5-27B | 完整 observation | 无 |
| `ir_structural_keep50` | 相同 | IR/identifier/结构锚点，保留 50% 行 | 无 |
| `execution_ast_keep50` | 相同 | traceback/test/diff/grep/AST 规则，保留 50% 行 | 无 |
| `ir_ast_hybrid_keep50` | 相同 | 两种 rank percentile 固定融合，保留 50% 行 | 无 |

四组使用同一个 vLLM endpoint、同一个 agent base config、相同 dataset、
split、slice/filter 和 worker 数。baseline 通过 agent 自己的
`--disable-pruner` 关闭剪枝。

本项目 conda 与 mini-swe-agent 环境可以分离。`MINI_EXTRA_BIN` 指向已有
pruning 版 agent 的绝对路径，脚本使用其同环境 Python 做 contract
校验；pruner 服务仍由本项目 conda 运行。

## 启动阶段

零参数 launch 默认运行 SWE-Bench Verified test 前 10 题：

```bash
bash scripts/run_server_experiments.sh
```

真实一题烟测使用：

```bash
bash scripts/run_server_experiments.sh smoke
```

`smoke` 仍会启动 mini-swe-agent、Docker 和 vLLM 推理；只是固定
`TASK_SLICE=0:1` 且比较 baseline/IR。它不是本仓库 demo JSONL。

## 多实验

单一 keep ratio 用于先验证链路。正式预算曲线可以一次生成：

```bash
KEEP_RATIOS=0.25,0.35,0.5,0.7,1.0 \
TASK_SLICE=0:50 \
bash scripts/run_server_experiments.sh
```

注意 `keep_ratio=1.0` 仍使用带 context-focus prompt 的 pruning arm，
适合控制 prompt/protocol 变化；真正的原始 baseline 仍是独立
`baseline`。

旧 `threshold` 字段只承担兼容传输：配置适配器固定使用
`threshold=1-keep_ratio`。它不是训练模型概率，manifest 会记录该语义。

默认所有 arm 并行，共享 vLLM。研究端到端延迟时建议
`PARALLEL_ARMS=0`，避免不同 arm 的排队互相污染。

## 结果含义

`results` 从 mini-swe-agent trajectory 统计：

- 完成 trajectory 和生成 prediction 的数量；
- `Submitted` 数量（只表示 agent 提交 patch，不表示通过测试）；
- vLLM API calls；
- prompt、completion 和 total tokens；
- 每个 arm 的 wall time；
- 有 query 的 pruner 调用次数；
- 原 observation token 与保留 token；
- pruner error 次数。

只有 `grade` 运行官方 SWE-Bench harness 后才报告：

- resolved task 数；
- graded task 数；
- Resolve Rate。

因此 `summary.csv` 中的空 Resolve Rate 是“尚未评分”，不是 0，也不会用
离线 macro line recall 替代。

## 公平性与当前限制

- Qwen temperature 固定为 0，但并行 vLLM 调度仍可能影响 wall time；
- 官方 pruning-capable mini-swe-agent 只在模型产生
  `context_focus_question` 时调用 pruner，因此 `prune_calls` 必须检查；
- 当前 client 只发送 `query/code/threshold`，AST 方法从 observation
  内容自动识别 traceback/diff/test/grep/source 类型；
- IR/AST/hybrid 不调用额外模型，`model_forward_count=0`；
- PPL/hidden/attention/influence 不应伪装成这三组 online CPU 方法，
  它们需要单独的本地模型或 serving signal 集成。
- 原始/保留 observation token 是兼容服务的轻量估算；端到端
  prompt/completion token 以 vLLM trajectory usage 为准。
