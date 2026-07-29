# B200 服务器引导

本项目沿用现有服务器约定：

```text
仓库：/home/yuantao/futao/swepruner_training_free_workspace/swe-pruner-training-free
conda：swepruner-training-free
Python：3.11+
GPU：B200
vLLM：http://127.0.0.1:8015/v1
Agent：服务器已经安装的 mini-swe-agent
```

## 真实实验的数据流

```text
SWE-Bench Verified issue
  -> mini-swe-agent
  -> qwen3.5-27b on the existing vLLM server
  -> shell command inside the SWE-Bench Docker image
  -> raw shell observation
  -> local training-free /prune service
  -> pruned observation returned to mini-swe-agent
  -> patch + trajectory + preds.json
  -> official SWE-Bench grader
```

`scripts/run_server_experiments.sh` 只适配已安装的 agent，不安装、复制或
重写 mini-swe-agent。它要求现有版本已经提供 `--pruner-url`、
`--disable-pruner` 和 `context_focus_question` pruning hook；不满足时
preflight 会直接失败。适配契约见 `docs/MINI_SWE_AGENT_ADAPTER.md`。

## 一键入口

在已经拉取的仓库中：

```bash
git pull --ff-only
bash scripts/run_server_experiments.sh preflight
bash scripts/run_server_experiments.sh
```

脚本会在自己的子进程中先清理当前 uv/venv，再激活 conda。它会：

1. 请求 `http://127.0.0.1:8015/v1/models`，优先选择 id 中含
   `qwen3.5` 的模型；
2. 检查 Docker daemon 和 mini-swe-agent pruning hook；
3. 以 localhost 端口 `8111/8112/8113` 启动 IR、AST、IR+AST 服务；
4. 在完全相同的 SWE-Bench Verified task slice 上启动 baseline 和三个
   training-free pruning arm；
5. 为每组保存独立的 config、PID、日志、trajectory 和 `preds.json`。

默认 `TASK_SLICE=0:10`，也就是真实 benchmark 的前 10 题。不存在 replay
或 demo 回退。

## 查看与评分

```bash
bash scripts/run_server_experiments.sh status
bash scripts/run_server_experiments.sh results
bash scripts/run_server_experiments.sh grade
bash scripts/run_server_experiments.sh stop
```

- `results` 汇总 task 数、API calls、prompt/completion token、wall time、
  剪枝调用、观测 token 保留率和错误数；
- `grade` 调用本机安装的 `swebench.harness.run_evaluation`，然后把官方
  report 中的 Resolve Rate 合并回 `summary.csv/json`；
- grader 未运行时，Resolve Rate 保持空值；
- `stop` 只处理该 run 目录中记录且命令行匹配的 agent/pruner PID。

结果目录：

```text
$WORK_DIR/agent_runs/<run_tag>/
├── manifest.json
├── configs/
├── services/
├── arms/
│   ├── baseline/
│   ├── ir_structural_keep50/
│   ├── execution_ast_keep50/
│   └── ir_ast_hybrid_keep50/
├── grade/
├── summary.csv
└── summary.json
```

## 多预算与规模化

环境变量都可以放在服务器自己的 job wrapper 中；仓库不会保存密钥：

```bash
TASK_SLICE=0:50 \
KEEP_RATIOS=0.35,0.5,0.7 \
AGENT_WORKERS=4 \
bash scripts/run_server_experiments.sh
```

全量 500 题时将 `TASK_SLICE` 设为空。若只跑部分方法：

```bash
METHODS=ir_structural,ir_ast_hybrid \
TASK_FILTER='^(django__django-|sympy__sympy-)' \
bash scripts/run_server_experiments.sh
```

默认 `PARALLEL_ARMS=1` 会让多组 agent 共享同一个 vLLM。若希望避免组间
吞吐干扰并获得更干净的 wall-time 数据，可设 `PARALLEL_ARMS=0` 串行
执行；temperature 固定为 0。

## vLLM model id 与认证

通常无需手填 model id。若 `/v1/models` 返回多个模型，可显式指定：

```bash
VLLM_MODEL_ID='Qwen/Qwen3.5-27B' \
bash scripts/run_server_experiments.sh preflight
```

本地 vLLM 默认使用占位 key `EMPTY`。如果服务启用了鉴权，只在服务器
环境中设置 `VLLM_API_KEY`，不要写入 config 或提交 Git。

## 离线方法仍然保留

`scripts/run_replay_matrix.sh` 仍用于算法级 line recall/critical-miss
排查；`scripts/run_server_experiments.sh` 专门用于真实 agent benchmark。
两者输出和指标不可混称。hidden-state、attention、PPL、influence 等
需要额外推理信号的路线仍按 `docs/EXPERIMENT_GUIDE.md` 做第二阶段实验。
