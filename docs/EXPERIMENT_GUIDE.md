# 实验引导

## 1. 先做离线 replay

从真实 coding-agent 轨迹抽取：

```text
(history, tool_call, tool_response, next_action)
```

至少保留 `request_id`、tool 类型、路径、query、原始 response、下一步 action，以及可人工检查的关键行。不要先把 benchmark 全量跑起来；先用 50–200 条长观测定位失败模式。

当前父项目的结构化数据可以直接转换：

```bash
python -m evaluation.convert_existing \
  --input ../artifacts/combined_2k/pruning_sft.jsonl \
  --output outputs/replay_200.jsonl \
  --limit 200 \
  --required-confidence 0.9
```

`line_keep_labels=1` 会转为 `gold_line_numbers`；其中置信度至少 0.9
的行会转为更严格的 `required_line_numbers`。转换使用代码片段内的相对
行号，不会误用原仓库文件的绝对行号。

父项目的行级训练数据没有真实 trajectory 的 `next_action`，因此可用于
IR/PPL/hidden/attention/AST 排序评测，但不能凭空作为 influence oracle
的目标。Influence replay 必须从真实 agent 轨迹把下一步 action 写入
`request.metadata.next_action`。

## 2. 固定预算矩阵

所有方法都跑：

```text
100%, 70%, 50%, 35%, 25%
```

也应单独比较 `configs/length_aware_budget.json`。固定比例用于公平作图，长度预算用于接近实际部署。

```bash
bash scripts/run_replay_matrix.sh \
  METHOD \
  /path/to/replay.jsonl \
  outputs/METHOD
```

## 3. 主实验顺序

1. `ir_structural`：验证低成本、闭源可用的代码感知基线。
2. `hidden_state_similarity`：验证不训练 head 是否能直接读取 backbone 相关性。
3. `influence_oracle`：在小样本上估计 likelihood-preserving 删除上界。
4. 用 `execution_ast` 与前三项做规则回退或 rank fusion。
5. 再比较 `conditional_ppl` 与 `attention_rollout` 的额外收益/系统开销。

## 4. 必做消融

- query：Goal Hint / tool call / error+identifier 自动提取；
- 预算：固定比例 / 长度感知；
- 粒度：行 / block / 函数；
- tool：source / grep / traceback / test log / diff；
- IR：窗口、结构扩张、权重；
- hidden：layer、pooling、anchor 来源；
- attention：首步/前 3/前 5 decode，最后 4 层/全层；
- influence：block 数、objective、是否分层。

## 5. 进入真实 agent

离线筛掉 Pareto 明显劣势的方法后，再按以下顺序：

1. SWE-QA / SWE-QA-Pro 小子集；
2. SWE-Bench Verified 100–200 条；
3. 全量核心 benchmark；
4. Oolong 与 Long Code QA/Completion 作为稳健性与结构副作用评测。

真实 agent 需要同时记录：

- Resolve Rate、Judge Score 或 Accuracy；
- 端到端输入/输出 token；
- API calls；
- wall time 与 TTFT；
- 每类 tool 的 critical miss；
- AST correctness 或等价结构指标。

当前仓库已经为第一阶段提供
`scripts/run_server_experiments.sh`：连接端口 8015 的 Qwen3.5 vLLM 和
已安装 mini-swe-agent，比较 no-pruning、IR、AST、IR+AST，并调用官方
SWE-Bench grader。详见 `docs/CODING_AGENT_EXPERIMENTS.md`。

## 6. 建议成功门槛

- SWE-QA / SWE-QA-Pro：相对 No Pruning 的 judge score 降低不超过 0.15–0.25，同时端到端 token 降低至少 20%；
- SWE-Bench Verified：Resolve Rate 下降不超过 1–2 个点，同时 token 或 wall time 至少一项改善；
- in-engine 方法：额外 wall time 目标不超过约 15%；
- traceback/diff/grep 的定位行必须单独报告漏保率。

这些是筛选门槛，不是代码单元测试通过就能证明的结论。
