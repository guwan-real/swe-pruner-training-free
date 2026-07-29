# B200 服务器引导

本项目沿用现有项目环境，不改变服务器：

```text
基础目录：/home/yuantao/futao
Python：3.11
GPU：B200
PyTorch：CUDA 13.0 wheel
```

## 1. clone 新 repo

```bash
export BASE_DIR=/home/yuantao/futao
export WORK_DIR=$BASE_DIR/swepruner_training_free_workspace
export PROJECT_DIR=$WORK_DIR/swe-pruner-training-free

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
git clone https://github.com/guwan-real/swe-pruner-training-free.git \
  swe-pruner-training-free
cd "$PROJECT_DIR"
```

## 2. 创建 conda 环境

```bash
bash scripts/create_server_conda.sh swepruner-training-free
conda activate swepruner-training-free
```

脚本默认先安装：

```text
torch==2.12.1
index=https://download.pytorch.org/whl/cu130
```

如服务器镜像变化，可显式覆盖：

```bash
TORCH_VERSION=2.12.1 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
bash scripts/create_server_conda.sh swepruner-training-free
```

## 3. 一键启动并行实验

环境安装完成后不需要手工复制多条实验命令：

```bash
cd /home/yuantao/futao/swepruner_training_free_workspace/swe-pruner-training-free
git pull --ff-only
bash scripts/run_server_experiments.sh
```

新终端即使默认激活了 uv 项目环境也可以直接运行。脚本会在自己的
子进程中：

1. 从 `PATH` 移除当前 `VIRTUAL_ENV/bin` 并清理 uv/venv 环境变量；
2. 激活 `swepruner-training-free` conda 环境；
3. 优先复用 `$WORK_DIR/replay/replay_200.jsonl`；
4. 若 replay 不存在，则从旧项目 `artifacts/combined_2k/pruning_sft.jsonl`
   转换最多 200 条；
5. 若旧数据也不存在，则明确告警并回退到仓库自带 demo；
6. 用 `nohup` 并行启动 IR 与 AST 的五档预算矩阵。

脚本只改变自己的子进程环境，不会修改调用它的父终端。查看进度和结果：

```bash
bash scripts/run_server_experiments.sh status
bash scripts/run_server_experiments.sh results
```

只做前台烟测：

```bash
bash scripts/run_server_experiments.sh smoke
```

输出分别位于：

```text
$WORK_DIR/runs/<run_tag>/
$WORK_DIR/logs/<run_tag>/
```

以下可选输入存在时，一键脚本会自动增加对应实验：

| 方法 | 自动发现的输入 |
|---|---|
| conditional PPL | `$WORK_DIR/local_configs/conditional_ppl.json` |
| hidden-state | `$WORK_DIR/replay/hidden_signals.jsonl` |
| attention | `$WORK_DIR/replay/attention_signals.jsonl` |
| influence oracle | `local_configs/influence_oracle.json` 和 `replay/oracle_50.jsonl` |

PPL 默认使用物理 GPU 0，influence 默认使用物理 GPU 1；可分别通过
`PPL_GPU`、`INFLUENCE_GPU` 覆盖。运行
`bash scripts/run_server_experiments.sh --help` 可查看全部环境变量。

## 4. 模型与数组目录

推荐：

```text
$WORK_DIR/
├── models/
│   └── LOCAL_CAUSAL_MODEL/
├── replay/
├── signals/
│   ├── hidden_states/
│   └── attention/
├── runs/
└── logs/
```

配置中写服务器本地路径。HF 加载器固定 `local_files_only=True`；如模型尚未下载，应先用服务器现有下载流程准备，不能让实验运行时静默联网。

用冻结模型导出两种 Pro 路线需要的信号：

```bash
python scripts/extract_hf_signals.py \
  --model-path "$WORK_DIR/models/LOCAL_CAUSAL_MODEL" \
  --request examples/requests/source_file.json \
  --output "$WORK_DIR/signals/source_file.npz" \
  --signals both \
  --device cuda \
  --dtype bfloat16 \
  --hidden-layers 4 \
  --decode-steps 3
```

输出的 hidden shape 是 `[layers,tokens,hidden]`，attention mass shape 是
`[layers,heads,decode_steps,tokens]`；`token_to_line` 使用 1-based 行号。
同时生成的 `source_file.request.json` 已写入本地 NPZ 路径，可直接运行。

## 5. 指定 GPU

先验证与现有 tool wrapper 兼容的 CPU 服务：

```bash
tf-prune-serve \
  --method ir_structural \
  --config tasks/ir_structural/config.example.json \
  --host 127.0.0.1 \
  --port 8000 \
  --no-prune-below 20
```

另一个终端检查：

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/prune \
  -H 'Content-Type: application/json' \
  --data '{"query":"find timeout","code":"line one\nDEFAULT_TIMEOUT = 30"}'
```

服务默认 fail-open。只有离线故障测试才建议加 `--fail-closed`。标准库
HTTP server 没有认证/TLS，默认不要改成公网监听地址。

```bash
export CUDA_VISIBLE_DEVICES=0
python -m tf_pruning.cli evaluate \
  --method hidden_state_similarity \
  --config tasks/hidden_state_similarity/config.example.json \
  --input "$WORK_DIR/replay/dev.jsonl" \
  --output-dir "$WORK_DIR/runs/hidden_dev" \
  --budget-schedule configs/length_aware_budget.json
```

Influence oracle 会执行多次 forward，建议单 GPU、小样本、较粗 block 先跑：

```bash
export CUDA_VISIBLE_DEVICES=1
python -m tf_pruning.cli evaluate \
  --method influence_oracle \
  --config /path/to/local_influence_config.json \
  --input "$WORK_DIR/replay/oracle_50.jsonl" \
  --output-dir "$WORK_DIR/runs/influence_oracle_50" \
  --keep-ratio 0.5 \
  --no-prune-below 0
```

## 6. 并行原则

- IR/AST 用 CPU，不占 B200；
- hidden/attention 优先消费推理引擎已经导出的数组，避免额外 forward；
- PPL 与 influence 分配不同物理 GPU；
- 日志、runs、signals 放在 workspace，不提交 Git；
- 先完成 smoke 和 50 条 replay，再启动大规模 benchmark。
