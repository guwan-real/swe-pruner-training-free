# 研究方案到代码的映射

本文档用于审计 `deep-research-report.md` 中的六条候选路线是否都已有对应代码。这里的“完成”指本地算法、接口、测试与运行引导完成，不代表已经在 SWE-Bench/SWE-QA 上得到论文结论。

## 两个替换位置

| 基线替换位置 | 本项目实现 | 保留的基线能力 |
|---|---|---|
| SWE-Pruner A3：外部 skimmer/scorer | IR、PPL、执行信号+AST | tool wrapper、Goal Hint/query、行级返回 |
| SWE-Pruner Pro B3：hidden-state pruning head | hidden similarity、attention | prefill/decoder tensors、行边界、骨架响应 |
| 离线上界/伪标签 | influence oracle | history/observation/next action replay |

## 1. IR + 结构锚点

- 代码：`tasks/ir_structural/pruner.py`
- 主类：`IRStructuralPruner`
- 已实现：内存 BM25、identifier overlap、path bonus、recent-context bonus、结构/错误锚点、上下文窗口扩张、硬行预算。
- 训练依赖：无。
- 模型调用：无。
- 测试：`tests/test_ir_structural.py`。

混合分数是固定 test-time 配置，不经过拟合：

```text
score(line) =
  w_bm25 * BM25(query, line_window)
  + w_id * identifier_overlap
  + w_path * path_overlap
  + w_recent * recent_overlap
  + structure/error/window bonuses
```

## 2. 条件 PPL + 首 token surprisal

- 代码：`tasks/conditional_ppl/pruner.py`
- 主类：`ConditionalPPLPruner`
- 冻结模型适配器：`HFConditionalSurprisalScorer`
- 已实现：code-aware block 粗排、完整 block mean-NLL、top block 行级首-token surprisal、结构/错误/显式 anchor 保护。
- 训练依赖：无；HF 权重只做 inference。
- 模型调用：每个 coarse block 一次，加每条被 refine 的行一次；结果中记录 `model_forward_count`。
- 测试：`tests/test_conditional_ppl.py` 使用 deterministic scorer，不下载模型。

## 3. Hidden-state 锚点相似度

- 代码：`tasks/hidden_state_similarity/pruner.py`
- 主类：`HiddenStateSimilarityPruner`
- 已实现：`[tokens,hidden]` / `[layers,tokens,hidden]`、token-to-line、mean/max/last/last-4 pooling、query/tool/error/decode 四类 anchor、余弦融合、NPZ 输入。
- 训练依赖：无 pruning head、无 probe、无拟合。
- 模型调用：pruner 本身为 0；线上应复用 serving prefill states。
- 测试：`tests/test_hidden_state_similarity.py`。

本地冻结 HF 导出器位于 `scripts/extract_hf_signals.py`；它用于算法实验，不代表生产 serving 集成。

## 4. Attention rollout / heavy hitter

- 代码：`tasks/attention_rollout/pruner.py`
- 主类：`AttentionRolloutPruner`
- 已实现：decode attention mass、方阵 attention rollout、layer/head/step 选择与聚合、step decay、attention sink 屏蔽、Top-P、硬预算、结构/局部 floor。
- 输入：默认 mass shape `[layers,heads,steps,tokens]`；rollout 接受 `[layers,heads,queries,keys]`。
- 训练依赖：无。
- 模型调用：pruner 本身为 0；线上应复用 decode attention。
- 测试：`tests/test_attention_rollout.py`。

## 5. 影响函数式贪心删除

- 代码：`tasks/influence_oracle/pruner.py`
- 主类：`InfluenceOraclePruner`
- 冻结模型适配器：`HFLogLikelihoodScorer`
- 已实现：code-aware disjoint blocks、leave-one-block-out、hierarchical greedy、精确行预算 refine、next-action log-likelihood objective、cache 与 forward 上限。
- 训练依赖：无；HF 权重只做 inference。
- 模型调用：高，结果中准确记录 `model_forward_count`。
- 定位：`small-sample-offline` oracle，不作为第一代线上默认。
- 测试：`tests/test_influence_oracle.py`。

## 6. 执行信号 + AST 骨架

- 代码：`tasks/execution_ast/pruner.py`
- 主类：`ExecutionASTPruner`
- 已实现：tool auto detection；source/grep/traceback/diff/test log/tree/generic 独立规则；Python import/声明骨架；命中符号定义体；一跳 caller/callee 展开；显式 mandatory 与上下文行。
- 训练依赖：无。
- 模型调用：无。
- 测试：`tests/test_execution_ast.py`。

## 共享实验要求

| 报告要求 | 代码证据 |
|---|---|
| 预算 `{100,70,50,35,25}%` | `scripts/run_replay_matrix.sh` |
| 长度感知预算 | `tf_pruning/budgets.py`、`configs/length_aware_budget.json` |
| line recall/F1、required recall | `evaluation/metrics.py` |
| token、latency、forward 成本 | `evaluation/metrics.py` |
| tool 类型分层 | `aggregate_metrics(...).by_tool_type` |
| 离线 replay | `evaluation/replay.py` |
| 父项目数据复用 | `evaluation/convert_existing.py` |
| Pareto 汇总 | `evaluation/matrix.py` |
| B200/Python 3.11 环境 | `scripts/create_server_conda.sh`、`docs/SERVER_GUIDE.md` |
| 现有 tool wrapper 接入 | `integrations/middleware.py`、`integrations/http_server.py` |

## 尚未声称完成的外部阶段

以下事项不属于本仓库当前的本地算法正确性结论：

- 下载/挂载服务器本地模型；
- 接 patched SGLang 或其他 serving engine；
- 采集真实 agent trajectory；
- 运行 SWE-QA、SWE-QA-Pro、SWE-Bench Verified、Oolong 或 Long Code 全量实验；
- 根据 benchmark 结果调整 rank fusion 与回退阈值。

因此，当前实现证明的是“六条路线可编码、可独立调用、可统一 replay”，
不把单元测试或服务器烟测结果冒充 benchmark 质量结论。
