# 服务器本地 Agent 交接说明

本文档写给运行在 B200 服务器上的本地 coding agent。目标是让它先完成
环境审计和一题 smoke，再决定是否扩大实验；不要重新实现 mini-swe-agent，
不要启动新的 vLLM，也不要把离线 replay 当成 agent benchmark。

## 1. 已知事实与目标

已知服务器状态：

- 项目仓库：
  `/home/yuantao/futao/swepruner_training_free_workspace/swe-pruner-training-free`
- vLLM 已由用户启动，OpenAI-compatible API：
  `http://127.0.0.1:8015/v1`
- 目标模型：Qwen3.5-27B；实际 served model id 以 `/v1/models` 为准
- 服务器已有 `mini-swe-agent--with-pruning`
- 原 SWE-Pruner 服务可能仍在 `8016`，新服务使用 `8111/8112/8113`，
  两者不冲突
- Docker 用于 SWE-Bench task environment
- 磁盘曾经接近满载，因此必须检查 workspace 和 Docker root filesystem

本轮目标：

```text
SWE-Bench Verified task
  -> existing mini-swe-agent--with-pruning
  -> existing Qwen3.5-27B vLLM :8015
  -> shell observation
  -> baseline / IR / AST / IR+AST
  -> trajectory + patch
  -> official SWE-Bench grader
```

## 2. 三个环境必须分清

不要强行把所有组件装进一个环境。

| 组件 | 推荐环境 | 用途 |
|---|---|---|
| `swepruner-training-free` conda | 本仓库轻量环境 | HTTP pruner、配置适配、结果汇总 |
| 现有 mini-swe-agent venv/uv | 保持原样 | `mini-extra swebench` 和 pruning hook |
| SWE-Bench harness 环境 | 可与上面任一环境相同，也可独立 | `swebench.harness.run_evaluation` |

启动脚本会先记录当前 `mini-extra` 的绝对路径，再清理继承的 uv/venv，
然后激活本项目 conda。之后 agent arm 仍通过那个绝对路径运行，所以
mini-swe-agent 不需要重复安装到 conda。

如果轻量 conda 尚未创建，运行
`scripts/create_server_conda.sh`。它默认 `PROFILE=agent`，不安装
PyTorch/Transformers，不占用额外模型磁盘。只有第二阶段 PPL/hidden/
attention/influence 才使用 `PROFILE=model`。

## 3. 首次只生成本地 profile

本地 agent 应先定位以下真实文件：

1. pruning 版 `mini-extra` 可执行文件；
2. 与它同一 venv 的 Python；
3. `mini-swe-agent--with-pruning` 源码根目录；
4. 含 `agent.pruner` 与 `context_focus_question` 的 YAML；
5. 可选：装有 SWE-Bench harness 的 Python。

把 `configs/server_profile.example.env` 复制为仓库根目录的
`server_profile.env` 并填写真实绝对路径。该文件已被 `.gitignore`
排除，不得提交。

典型 profile 字段：

```text
MINI_EXTRA_BIN=/.../bin/mini-extra
MINI_SWE_PYTHON=/.../bin/python
MINI_SWE_AGENT_ROOT=/.../mini-swe-agent--with-pruning
MINI_SWE_BASE_CONFIG=/.../templates/.../pruner_local_qwen35_...yaml
SWEBENCH_PYTHON=/.../bin/python
```

如果不填 `MINI_SWE_BASE_CONFIG`，脚本先检查该 mini 安装自带的
`config/extra/swebench.yaml`。若它不兼容，则在
`MINI_SWE_AGENT_ROOT/templates` 中查找：

- 恰好一个兼容 YAML：自动使用；
- 多个兼容 YAML：拒绝猜测，并列出候选，要求明确指定。

## 4. preflight 必须通过

本地 agent 的第一项动作只能是：

```bash
bash scripts/run_server_experiments.sh preflight
```

preflight 会依次验证：

1. 本项目 conda 和 Python 3.11+；
2. 现有 `mini-extra` 的实际绝对路径及其专属 Python；
3. `--pruner-url`、`--disable-pruner`、`--slice`；
4. `PrunerRequest(query, code, threshold)`；
5. `PruneResponse(pruned_code, origin_token_cnt, left_token_cnt,
   model_input_token_cnt)`；
6. base YAML 的 `agent.pruner` 与 `context_focus_question`；
7. Docker daemon；
8. workspace 与 Docker filesystem 的剩余空间；
9. `8015/v1/models` 及实际 Qwen model id；
10. 官方 grader Python（缺少时只告警，不阻止生成 trajectory）。

磁盘规则：

- 少于 10GB：preflight 失败；
- 10–50GB：明确警告，可运行一题 smoke；
- 大于等于 50GB：正常；
- 扩大到 10/50/500 题前，必须重新评估 Docker image 占用。

若 preflight 失败，只修复它报告的前置条件。不要修改 pruner 算法，不要
用 toy replay 绕过检查。

## 5. smoke 的准确含义

preflight 通过后运行：

```bash
bash scripts/run_server_experiments.sh smoke
```

smoke 固定：

- `SWE-Bench Verified/test`
- `TASK_SLICE=0:1`
- baseline：`--disable-pruner`
- IR：`--pruner-url http://127.0.0.1:8111/prune`
- IR hard keep ratio：`0.5`
- 同一个 Qwen vLLM、同一个 task、同一个 agent base config

这是真实 agent/tool/Docker/model loop，不是 `examples/replay/demo.jsonl`。

参数转发由 `scripts/run_one_agent_arm.sh` 使用 `"$@"` 完整保留；带空格
的 filter/config/path 已有自动测试，不要自行改写成字符串拼接。

## 6. smoke 验收

本地 agent 应检查：

1. `status` 中 baseline 和 IR 都是 `completed`；
2. 两个 arm 都有 `preds.json` 和 `.traj.json`；
3. IR trajectory 中 `prune_calls > 0`；
4. `prune_errors == 0`；
5. `original_observation_tokens > kept_observation_tokens`；
6. baseline 的 pruner 统计为空；
7. `runner_exit_code == 0`；
8. `grade` 后 `resolved/graded/resolve_rate` 不再为空。

`Submitted` 只表示 agent 交了 patch，不代表 task resolved。只有官方
grader 的 Resolve Rate 才能作为 SWE-Bench 成功率。

## 7. threshold 语义

mini-swe-agent 仍发送旧字段 `threshold`，但 training-free ranker 使用
hard line budget。配置适配器固定写入：

```text
mini threshold = 1 - keep_ratio
```

例如 `KEEP_RATIOS=0.5` 会生成 `threshold=0.5`。这是兼容映射，不是
学习模型的概率阈值。每个 run 的 `manifest.json` 会记录该语义。

## 8. 扩大实验的门禁

只有一题 smoke 和官方 grade 均完成后，才允许：

1. 默认 10 题：baseline、IR、AST、IR+AST；
2. 50 题：先固定 keep ratio 0.5；
3. 多预算：0.25/0.35/0.5/0.7/1.0；
4. 最后才考虑 Verified 全量。

测 wall time 时使用 `PARALLEL_ARMS=0`，避免多个 arm 共享 vLLM 时互相
排队。比较质量时所有 arm 必须使用同一 task slice/filter。

## 9. 本地 Agent 最终回报格式

完成 preflight 或 smoke 后，只回报以下事实：

```text
resolved mini-extra:
resolved mini Python:
resolved base config:
resolved vLLM model id:
workspace free GB:
Docker free GB:
grader Python:
run root:
baseline state/tasks/API calls:
IR state/tasks/API calls/prune calls/retention/errors:
official resolved/graded:
next blocker:
```

不要报告 API key，不要粘贴完整私有 trajectory，不要把离线 line recall
写成 Resolve Rate。
