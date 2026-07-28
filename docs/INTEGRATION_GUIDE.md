# Coding-agent 集成引导

## SWE-Pruner 外挂路线

保留现有 tool wrapper 与 `context_focus_question`：

```text
tool response
  -> 构造 PruningRequest
  -> ir_structural / conditional_ppl / execution_ast
  -> PruningResult.pruned_text
  -> 返回给 agent
```

替换点只在原 skimmer/scorer，不修改 agent 的工具协议。closed-source API 优先使用这条路线。

## SWE-Pruner Pro 内嵌路线

保留推理引擎导出 hidden states/attention 的能力：

```text
prefill tool response
  -> token states + token_to_line
  -> hidden_state_similarity 或 attention_rollout
  -> 行级 Top-P/预算
  -> 下一轮使用骨架响应
```

替换的是训练过的 pruning head，不是 hidden-state 提取链。优先离线保存 `.npz` 验证排序，再接 patched serving engine，避免同时改变算法和系统。

`scripts/extract_hf_signals.py` 是本地 Hugging Face 离线适配器，用于在
接推理引擎前验证数据 shape、行映射与算法。它不是 patched SGLang 的
生产替代：线上实验应直接复用 prefill/decode 已产生的 tensors，避免
为了剪枝再做一次完整 forward。

## 通用适配器

每次 tool 完成后：

1. 原始字符串写入 `PruningRequest.text`；
2. tool 名和参数映射到 `tool_type`、`path`；
3. Goal Hint、用户问题、报错与最近符号写入 `query/recent_context`；
4. 用长度预算生成 `BudgetConfig`；
5. 调用对应 `Pruner`；
6. 把 `pruned_text` 放回 observation，同时把原始文本与 result 写入 replay 日志；
7. 若 pruner 失败、信号缺失或结果违反保底规则，返回原始 response。

上述流程已实现为 `integrations.middleware.TrainingFreeMiddleware`：

```python
from integrations.middleware import TrainingFreeMiddleware
from tf_pruning.registry import build_pruner

middleware = TrainingFreeMiddleware(
    build_pruner("ir_structural"),
    fail_open=True,
)
outcome = middleware.prune_tool_response(
    tool_output,
    query=context_focus_question,
    tool_name="read_file",
    path="src/client.py",
)
observation_for_agent = outcome.text
```

middleware 会根据长度选择预算，并从 tool/command/path 推断
source、grep、traceback、diff、test log 或 tree。默认 fail-open。

## 官方 HTTP 兼容层

现有 SWE-Pruner 示例通过 `PRUNER_URL=http://localhost:8000/prune`
发送 `query/code/threshold`。本项目可直接提供相同核心字段：

```bash
tf-prune-serve \
  --method ir_structural \
  --config tasks/ir_structural/config.example.json \
  --host 127.0.0.1 \
  --port 8000
```

返回仍包含：

```text
score, pruned_code, token_scores, kept_frags,
origin_token_cnt, left_token_cnt, model_input_token_cnt, error_msg
```

注意：原版 `threshold` 是训练模型的概率阈值，training-free 方法使用
hard budget。兼容层只为旧 wrapper 做 `keep_ratio=1-threshold` 的单调
映射；可控实验应直接传 `keep_ratio` 或完整 `budget`。
`score` 是当前方法保留行的最大原生分数，不是跨方法可比较的概率；
扩展字段 `score_semantics` 与 `token_scores_granularity` 会明确这一点。

该服务使用 Python 标准库 HTTP server，默认仅监听 localhost，没有
认证或 TLS；不得直接暴露到公网。

## 必须保留的安全边界

- 短响应默认不剪；
- traceback 异常行/frame、diff hunk header、grep 命中、失败 assertion 设为 mandatory；
- 输出必须包含省略标记；
- 不把模型路径、API key 或完整私有轨迹写入 Git；
- pruner 异常不能让 agent tool call 失败；
- 线上前对每种 tool 单独校验 critical miss rate。

## Rank fusion

建议先做可解释的线性融合：

```text
final_score =
  primary_method_score
  + structural_floor
  + execution_signal_bonus
```

结构/执行信号作为保底，不应被纯相似度或 PPL 完全覆盖。融合权重是 test-time 配置，不通过训练获得；评测时必须单独报告单方法与融合方法，避免混淆收益来源。
