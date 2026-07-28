# 新 repo 发布检查清单

## 已确认的发布范围

- repo：`guwan-real/swe-pruner-training-free`；
- visibility：public；
- 不添加 License；
- 不提交研究报告原文、父项目数据或模型；
- 首个版本包含本地实现、测试、示例与部署引导。

## 建议仓库边界

以当前 `training_free_pruning/` 目录作为新 repo 根目录，不把父项目的
训练数据、官方 checkpoint、历史 artifacts 或旧 Git 记录带入。

严禁提交：

- `models/`、`.safetensors`、`.pt`、`.npz`；
- replay 中的私有轨迹；
- API key、token、服务器凭据；
- `outputs/`、`runs/`、pytest/cache/egg-info；
- 父项目 `artifacts/` 的 2K 数据包，除非另行确认数据许可与 repo 大小。

## 发布门禁

```bash
bash scripts/run_local_checks.sh
tf-prune-serve --help
python -m tf_pruning.cli prune \
  --method ir_structural \
  --request examples/requests/source_file.json
python -m tf_pruning.cli evaluate \
  --method execution_ast \
  --input examples/replay/demo.jsonl \
  --output-dir /tmp/execution_ast_publish_smoke \
  --keep-ratio 0.5 \
  --no-prune-below 0
```

还需检查：

```bash
git status --short
git diff --check
git grep -nE 'api[_-]?key|secret|token='
```

## 发布后服务器阶段

1. 在 `/home/yuantao/futao` 下 clone
   `https://github.com/guwan-real/swe-pruner-training-free.git`；
2. 按 `docs/SERVER_GUIDE.md` 创建独立 conda 环境；
3. 先跑 CPU 的 IR/AST smoke；
4. 挂载本地 causal model 与 replay；
5. 导出 hidden/attention signals；
6. 只在 50–200 条子集上筛选 Pareto；
7. 用户确认后再启动大规模 benchmark。
