# Training-free 后验动作剪枝方案

## 1. 这次实现解决的不是旧 `/prune` 问题

旧接口收到：

```json
{"query": "...", "code": "...", "threshold": 0.5}
```

此时 next action 还没有生成，所以它只能根据提前写出的
`context_focus_question`、IR、AST 或训练分类头的先验分数选片段。服务器
上简单模型给出的契约总结对这个旧流程是正确的，但不能表达本文的“动作
后验”。

新代码完全隔离在 `posterior_pruning/`，没有修改：

- `tf_pruning.protocol.Pruner`；
- `tf_pruning.registry.METHOD_MODULES`；
- `tasks/ir_structural`、`tasks/execution_ast` 或其他旧方法；
- 旧的 `integrations/http_server.py` 和 `/prune`。

新接口是：

```text
POST /prune-post-action
```

请求必须同时提供：

```json
{
  "messages": [{"role": "system", "content": "..."}, "..."],
  "observation_index": 3,
  "next_action": "已经由 agent 生成的完整 assistant response",
  "keep_ratio": 0.5,
  "query": "SWE-Bench problem statement",
  "request_id": "instance:turn"
}
```

服务只允许替换 `messages[observation_index]`，不会生成、改写或重新选择
`next_action`。

## 2. 后验分数怎样算

给定此前历史 \(H\)、完整 observation \(R\) 和 agent 已生成的动作
\(a=(a_1,\ldots,a_m)\)，冻结 Qwen 的动作分数是：

\[
S(R;a,H)=\frac{1}{m}\sum_{i=1}^{m}
\log p_\theta(a_i\mid H,R,a_{<i})
\]

候选压缩 observation 是 \(R'\)。定义平均 token log-probability 损失：

\[
\Delta(R')=S(R;a,H)-S(R';a,H)
\]

只有当：

\[
\Delta(R') \le \epsilon
\]

才接受 \(R'\)。默认 \(\epsilon=0.08\)，它是实验超参数，不是分类概率，
需要在固定开发集上做 0.02/0.05/0.08/0.12 消融。

这里的“后验”是指：选择证据时已经观察到实际动作 \(a\)。代码没有训练
分类头，也没有新参数：

- `theta` 就是端口 8015 已启动的 Qwen3.5-27B；
- 不读取或保存 checkpoint；
- 不反向传播；
- 不更新 vLLM；
- 每个候选只做固定 continuation 的 likelihood forward。

## 3. vLLM 如何做精确 fixed-action scoring

`posterior_pruning.scoring.VLLMActionScorer` 使用同一个 OpenAI-compatible
vLLM：

1. `POST /v1/tokenize` 对历史 messages 使用
   `add_generation_prompt=true`，得到 assistant 起始边界；
2. 把真实动作作为最后一条 assistant message；
3. `POST /v1/chat/completions` 使用
   `add_generation_prompt=false`、`prompt_logprobs=1`、
   `return_token_ids=true`；
4. 验证两个请求产生的 chat-template token 前缀完全一致；
5. 只累加 assistant 边界之后真实 token id 的 log probability。

边界不一致、字段缺失、上下文超长或 vLLM 请求失败时，线上服务默认
fail-open，保留完整 observation。`preflight` 会真实做一次小型 scoring
probe，不能只用 `/health` 假装兼容。

如果 agent 使用了非默认 chat-template 参数，例如
`enable_thinking=false`，必须在 profile 中设置完全相同的
`CHAT_TEMPLATE_KWARGS_JSON='{"enable_thinking":false}'`。前缀校验会阻止
静默错算。

vLLM 当前协议中的 `prompt_logprobs`、`prompt_token_ids` 与
`add_generation_prompt` 可在
[vLLM Chat Completion protocol](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/chat_completion/protocol.py)
核对；`/tokenize` 的 chat 请求定义在
[vLLM Tokenize protocol](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/serve/tokenize/protocol.py)。

## 4. 四个隔离方法

| 方法 | 每次可剪枝 turn 的额外 forward | 做什么 | 使用范围 |
|---|---:|---|---|
| `single_verify` | 2 | 完整 observation 与一个目标预算候选比较 | 最便宜主方法 |
| `budget_search` | 2–5 | 从小预算开始，找到第一个通过后验门的候选 | 自适应预算主方法 |
| `greedy_blocks` | 2–7，硬上限 | 逐个尝试删除低成本 proposal block，每次用后验验证 | 更细粒度主方法 |
| `block_influence` | 3–8，硬上限 | leave-one-block-out 计算每块对动作的影响，再验证组合候选 | 小样本 oracle |

候选 proposal 使用的 action 词项、错误栈、diff、源码位置和结构行只是
降低搜索空间的无训练启发式。最终能否删除由 frozen Qwen 的动作
likelihood 决定。`block_influence` 的组合候选还会做一次最终验证，不能
把独立 LOO 分数直接相加后未经检查上线。

四个方法分别位于独立子目录，共享的只有新后验协议、候选表示和 scorer；
它们不调用旧 IR/AST 类。

## 5. mini-swe-agent 的准确时序

新 adapter 的时序是：

```text
messages 中已有完整 observation R(t-1)
        |
        v
原 DefaultAgent.query(messages)
        |
        v
生成并记录真实 assistant action a(t)
        |
        v
adapter 调 /prune-post-action，固定 a(t) 比较 R(t-1) 与候选
        |
        v
只替换 messages 中的 R(t-1)
        |
        v
执行 a(t)，得到新的完整 R(t)
```

所以：

- `a(t)` 一定看过完整 `R(t-1)`；
- 压缩后的 `R(t-1)` 只影响 `a(t+1)` 及以后；
- 当前命令执行结果不会因为后验服务而变化；
- baseline 和所有后验组使用同一 agent YAML 与同一模型参数。

这与 SWE-Pruner Pro 描述的“当前 turn 看完整 response，压缩 response
进入后续历史”一致，但把训练好的 hidden-state classification head 换成了
冻结模型对真实动作的 counterfactual likelihood。SWE-Pruner Pro 的
训练 head 与时序可参见
[SWE-Pruner Pro repository](https://github.com/Ayanami1314/swe-pruner-pro)。

## 6. 成本与应该报告的指标

额外成本不能藏在 agent API call 中。trajectory 的每次
`posterior_pruned_stats` 记录：

- `model_forward_count`；
- `scoring_prompt_tokens`；
- `candidates_evaluated`；
- `full_action_mean_logprob`/`selected_action_mean_logprob`；
- 接受、拒绝、跳过和错误；
- observation 估算 token 保留率；
- 服务 latency。

正常请求中 `model_forward_count` 是实际 scorer call 数；如果某次 call
在 HTTP/vLLM 过程中报错，则按已发起的 forward attempt 计数，宁可略微
高估也不把失败请求的成本隐藏掉。

保留率按最终返回给 agent 的完整文本计算，显式省略标记也计入 token。
如果候选只是减少了行数、但加上标记后不比原文短，`single_verify` 和
`budget_search` 不会为它消耗模型 forward，也不会把它记成节省。

结果汇总必须同时报告：

1. SWE-Bench official Resolve Rate；
2. agent 原始生成 prompt/completion tokens；
3. posterior 额外 forward 与 scoring prompt tokens；
4. observation retention；
5. 总 wall time；
6. posterior error rate。

只报告 observation 省了多少 token 而不报告额外 forward，会高估方法收益。
比较纯 wall time 时设 `PARALLEL_ARMS=0`；比较质量时可以并行，让 vLLM
batch，但各组 wall time 不再可直接比较。

## 7. 实验门禁

顺序固定为：

1. `preflight`：真实检查 mini 主循环、Docker、`/tokenize` 与
   `prompt_logprobs`；
2. `smoke`：同一 Verified 题的 baseline + `single_verify`；
3. 检查 trajectory 中 posterior forward 大于 0、error 为 0；
4. 运行官方 grader；
5. 再跑前 10 题五组；
6. 之后才扩大 task slice 和 epsilon/keep-ratio 网格。

`Submitted` 不是 resolved；没有官方 grader 时 `resolve_rate` 必须为空。
