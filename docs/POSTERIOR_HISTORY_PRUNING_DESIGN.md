# 后验历史压缩：Read Once, Compact Later

## 目标

这个实现针对前一轮 zero-forward 实验暴露的问题：在工具输出刚产生时立即压缩，会让 agent 的下一步决策缺少信息，进而增加 API calls。后验历史压缩不替换当前 observation，而是在 Qwen 已经完整读过它、并正常生成下一步 action 后，才压缩它在更晚 prompt 中的临时历史视图。

硬约束：

- 不训练任何 pruning head；
- 不调用额外 vLLM / OpenAI / HTTP pruning 服务；
- 不改写 agent shell action；
- 不改写 canonical `agent.messages` 中的原始 observation；
- 最新 `POSTERIOR_HOT_OBSERVATIONS` 条工具 observation 永远全文保留；
- diff 永不压缩；无法安全选择时 fail-open 保留全文。

## 时序

```text
tool output O_t
  -> 作为完整 observation 加入 canonical history
  -> Qwen 正常读取完整 O_t，生成 action A_(t+1)
  -> 从 A_(t+1) 提取 command / context_focus_question / 有界 response 文本
  -> 用这些后验信号选择 O_t 的证据块
  -> 在它离开 hot window 后，仅在发送给模型的临时消息副本中替换 O_t
```

因此，剪枝不会影响模型第一次基于 `O_t` 决定下一步做什么。完整消息仍留在 agent 内存和 trajectory 中，便于审计；发送 vLLM 的 copied prompt 才包含压缩版本。

官方 SWE-Pruner fork 的实际边界是：

```text
DefaultAgent.query()
  -> self.model.query(self.messages)
  -> parse_action(response)
  -> self.add_message("assistant", **response)
```

适配器在这个 `model.query(messages)` 前创建临时 view，返回后再把正常生成的 assistant message 追加回全文 canonical history。它不复制或改写 `execute_action` 的执行语义。

## 后验选择器

选择器输入为造成 observation 的原始 command，以及其后真实发生的 action：

- 后续 shell command；
- `context_focus_question`，如果原 agent 已经生成；
- 有界的 assistant response 文本。

它优先使用精确标识符；词项回退只使用在当前 observation 中低频的词，避免 `is`、`where` 等自然语言词命中所有代码块。保留规则包含：

- source：函数/类/import 骨架、精确符号所在块和一跳邻域；
- traceback/test log：异常、frame、file:line、assertion 与邻域；
- search/tree：路径和显式错误；
- diff：永不压缩。

没有后验命中、保留率过高、收益不足、输出超过安全上限时，返回全文。

## 指标

每条工具 observation 在 canonical trajectory 中带有 `posterior_history_stats`，但临时 prompt renderer 会剥离这个元数据，避免它进入 OpenAI/vLLM messages。结果汇总包括：

- `posterior_compacted_observations`：有多少旧 observation 获得压缩视图；
- `history_prompt_compactions`：这些视图实际进入模型 prompt 的次数；
- `estimated_history_tokens_saved`：按本地确定性 token 代理累计的节省；
- agent prompt / total tokens、API calls、wall time、official resolve rate；
- 相对 baseline 的差值。

所有 pruner model-forward、pruner LLM-token 指标固定为零；这里没有 pruner 服务或模型调用。

## 实验原则

先顺序跑 `baseline + posterior_adaptive`，不要与其它 arm 并发争用 vLLM：

1. 先运行一题 smoke，确认 canonical history 全文、prompt view 仅压缩冷历史；
2. 再跑 5 题，要求 API calls 不高于 baseline 约 2%，并观察 prompt token；
3. 通过后再跑 20 题并官方评分；
4. `safe` 是更保守的 selector ablation，只有 replay 显示 `adaptive` 的后验匹配不稳定时再加入。

`MIN_INPUT_TOKENS` 只影响是否值得压缩冷历史；它不会改变当前 observation 是否完整进入下一轮，因此与旧的“提高门槛以避免 agent 行为扰动”不是同一个机制。
