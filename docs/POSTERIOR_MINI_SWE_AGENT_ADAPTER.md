# Existing mini-swe-agent 后验适配说明

## 审计结论

本实现对照了 SWE-Pruner `public` 分支中的
`downstream_eval/multi_turn/swebench/mini-swe-agent--with-pruning`。
审计对应 commit `96171b5f3ecaf89745cbeb436c8893b57f3400bd`；运行时
preflight 不依赖该 hash，而是再次检查实际安装版本的 method contract。
该版本的 `DefaultAgent` 当前流程是：

```python
step() -> get_observation(query())
query(): model.query(self.messages) -> add assistant message
get_observation(): execute action -> legacy _apply_pruner() -> add user observation
```

旧 `_apply_pruner()` 在命令执行后立即调用 `/prune`，只能使用当前 action
提前生成的 `context_focus_question`，并且下一动作从一开始看到的就是
已剪输出。这不是本实验所需的真实 next-action posterior。

## 新 adapter 做了什么

`posterior_pruning.mini_adapter.swebench` 是入口 wrapper。它：

1. 导入服务器已经安装的 mini-swe-agent；
2. 只在当前进程包装 `DefaultAgent.query(self)`；
3. 先执行原 query，让当前动作从完整 observation 生成；
4. 原 query 已把 assistant response 加入 messages 后，定位它前面的
   user observation；
5. 把历史、observation index 和完整 assistant response 发给新服务；
6. 只替换上一条 observation 的 `content`；
7. 把紧凑统计写入该 message 的 `posterior_pruned_stats`。

它不复制 agent class、不复制 SWE-Bench runner、不更改 `preds.json` 或
trajectory 保存逻辑。

## 与 pruning fork 和标准 mini 的兼容

wrapper 运行时检查：

- `DefaultAgent.query` 必须仍是 `query(self)`；
- `DefaultAgent.add_message` 必须存在；
- SWE-Bench CLI 必须有 subset/split/output/workers/config 参数。

对于 SWE-Pruner 的 pruning fork，runner 的 `main()` 有
`disable_pruner` 参数。wrapper 会自动追加 `--disable-pruner`，确保旧
`agent.pruner` 不会先改 observation。对于不含该参数的标准
mini-swe-agent，不追加未知参数。

共享 config adapter 也会删除 YAML 中的 `agent.pruner`。如果运行中仍
发现 `self.pruner_client` 非空，后验 hook 会明确失败，而不是把两个剪枝
方法串联。

## baseline 的公平性

baseline 也使用同一个 wrapper、同一个生成后的 `agent.yaml` 和同一个
Qwen 配置；区别只是它没有 `POSTERIOR_PRUNER_URL`，所以不会安装后验
hook。对于 pruning fork，两组都会关闭 legacy hook。因此 prompt、模型、
task slice、Docker 环境与 agent 代码保持一致。

## 失败语义

服务超时、HTTP 错误或返回字段错误时：

- 原 observation 保持不变；
- trajectory 写入 `status=client_error`；
- 当前动作和命令执行不受影响；
- 后续汇总把它计入 `posterior_errors`。

不要为了“跑完”而删除这条错误统计。posterior error 不为 0 的实验不能
作为正式结果。

## 独立验证

使用 mini-swe-agent 自己的 Python：

```bash
PYTHONPATH=/home/yuantao/futao/swepruner_training_free_workspace/swe-pruner-training-free \
  /path/to/mini-venv/bin/python \
  -m posterior_pruning.mini_adapter.preflight
```

这个命令只验证适配点，不调用 vLLM。完整 launcher 的 `preflight` 还会
额外做一次真实 action-likelihood probe。
