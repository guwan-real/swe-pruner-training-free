# B200 服务器本地 Agent：后验剪枝实验交接

这份文件写给服务器上的本地 coding agent。不要重新实现 mini-swe-agent，
不要启动第二个 vLLM，不要修改旧 IR/AST 实验。目标是运行隔离的
post-action posterior 实验。

## 1. 固定路径与已有服务

仓库应位于：

```text
/home/yuantao/futao/swepruner_training_free_workspace/swe-pruner-training-free
```

用户已启动：

```text
http://127.0.0.1:8015/v1
```

目标模型是 Qwen3.5-27B；served model id 必须以 `/v1/models` 返回值为
准。不要另起模型进程。`8016` 可能是官方 SWE-Pruner 服务，新后验服务
只使用 `8121`–`8124`，不会占用它。

## 2. 首次准备

```bash
cd /home/yuantao/futao/swepruner_training_free_workspace/swe-pruner-training-free
git pull --ff-only
bash scripts/create_server_conda.sh
cp configs/posterior_server_profile.example.env posterior_server_profile.env
```

编辑未跟踪的 `posterior_server_profile.env`，至少填写：

```text
MINI_SWE_PYTHON=/现有/mini-swe-agent/venv/bin/python
MINI_SWE_BASE_CONFIG=/现有/mini-swe-agent/swebench.yaml
```

`MINI_SWE_PYTHON` 可以来自标准 mini-swe-agent，也可以来自
SWE-Pruner 的 pruning fork。不要把 mini 重新安装到项目 conda。

如果新终端默认激活 uv，不需要手动复制一串 deactivate 命令；launcher
会先从当前环境解析并记住 mini Python，再清理 `VIRTUAL_ENV`/uv PATH，
然后激活 `swepruner-training-free` conda。

## 3. preflight

```bash
bash scripts/run_posterior_swebench.sh preflight
```

必须同时通过：

1. 项目 conda 可导入 `posterior_pruning`；
2. mini Python 可导入 `DefaultAgent` 与 SWE-Bench runner；
3. `DefaultAgent.query(self)` 适配点未变化；
4. Docker daemon 可用；
5. 仓库与可读的 Docker root filesystem 至少有 10GB 空间；
6. `8015/v1/models` 返回模型；
7. `POST /v1/tokenize` 支持 chat messages；
8. `POST /v1/chat/completions` 返回 `prompt_token_ids` 和
   `prompt_logprobs`；
9. chat-template prefix 完全一致。

第 7–9 项会真实占用一次很小的 Qwen forward。不能仅凭 `/health`
通过。

## 4. 一题 smoke

```bash
bash scripts/run_posterior_swebench.sh smoke
```

它启动两组相同的 SWE-Bench Verified/test 第 1 题：

- `baseline`：无旧 pruner、无后验 hook；
- `single_verify`：当前动作先看完整 observation，再做 full/candidate
  两次 likelihood scoring。

查看：

```bash
bash scripts/run_posterior_swebench.sh status
bash scripts/run_posterior_swebench.sh results
```

`result` 也作为 `results` 的兼容别名，不会再出现 unknown command。

smoke 验收：

- 两个 arm 的 `runner_exit_code=0`；
- 两组都有 `preds.json` 和 `.traj.json`；
- baseline 的 `posterior_model_forwards=0`；
- single 的 `posterior_model_forwards>0`；
- `posterior_errors=0`；
- trajectory 中每个被处理 observation 有
  `posterior_pruned_stats`；
- 运行 grader 前 `resolve_rate` 为空。

## 5. 五组并行实验

smoke 和 grade 通过后：

```bash
bash scripts/run_posterior_swebench.sh launch
```

默认是同一 Verified `0:10`、同一 config、同一 Qwen 的五组：

| arm | 端口 | 额外 forward 上界/可剪 turn |
|---|---:|---:|
| baseline | 无 | 0 |
| single_verify | 8121 | 2 |
| budget_search | 8122 | 5 |
| greedy_blocks | 8123 | 7 |
| block_influence | 8124 | 8 |

`PARALLEL_ARMS=1` 会同时启动五组，vLLM 可进行 batching。每组默认
`AGENT_WORKERS=1`，不要一开始同时把 arm 和 worker 都放大。并行结果可
比较质量，但 wall time 受共享队列影响。需要干净 latency 时设置：

```text
PARALLEL_ARMS=0
```

然后用新 `RUN_TAG` 重跑。

## 6. 正式评分

trajectory 完成后：

```bash
bash scripts/run_posterior_swebench.sh grade
bash scripts/run_posterior_swebench.sh results
```

如果 harness 不在 mini 环境，在 profile 设置：

```text
SWEBENCH_PYTHON=/path/to/swebench-env/bin/python
```

只有 official grader 的 `resolve_rate` 是任务成功率。`Submitted` 只表示
agent 交了 patch。

## 7. 成本检查

正式回报至少包含：

```text
run root:
git commit:
vLLM model id:
task slice:
arm:
agent calls/tokens:
posterior calls/model forwards/scoring prompt tokens:
accepted/rejected/skipped/errors:
observation retention:
wall time:
official resolved/graded/resolve rate:
```

如果 `posterior_scoring_prompt_tokens` 很大、保留率收益小，先减少
`METHODS` 或降低 `MAX_POSTERIOR_EVALUATIONS`，不要把额外推理成本从
报告中删掉。

## 8. 停止范围

```bash
bash scripts/run_posterior_swebench.sh stop
```

只终止当前 run 记录的 posterior agent/service PID。它不会停止 8015
vLLM、8016 官方 pruner 或其他用户进程。
