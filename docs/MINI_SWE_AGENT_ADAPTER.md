# mini-swe-agent 适配契约

本仓库不维护第二份 mini-swe-agent。端到端脚本只要求服务器现有安装满足
一个很小的 pruning contract。

## 自动检查的能力

`bash scripts/run_server_experiments.sh preflight` 会确认：

1. `mini-extra swebench --help` 包含 `--pruner-url`；
2. 同一命令包含 `--disable-pruner`，用于真正的 no-pruning baseline；
3. 支持 `--slice`，确保每个 arm 使用同一批 task；
4. SWE-Bench base YAML 中存在 `agent.pruner`；
5. agent prompt 包含 `context_focus_question`；
6. mini-swe-agent 的 pruner client 会向 `POST /prune` 发送
   `query/code/threshold`。

配置适配器 `agent_eval/config_adapter.py` 读取已安装 agent 自己的
SWE-Bench YAML，只覆盖：

- `model.model_name=hosted_vllm/<GET /v1/models 返回的 id>`；
- `model.model_kwargs.api_base=http://127.0.0.1:8015/v1`；
- temperature、timeout 和本地 API key；
- 当前实验组的 pruner URL 与 hard keep ratio。

它不会修改 site-packages，也不会复制 agent loop。

## preflight 失败时

如果现有 mini-swe-agent 是完全未带 pruning hook 的上游版本，仅通过
YAML 无法让 observation 自动经过 `/prune`。需要在现有安装中提供与
SWE-Pruner 版本等价的以下最小适配：

1. `AgentConfig` 增加可选 `pruner` 配置；
2. shell command 执行后、observation 写入 history 前调用 pruner client；
3. 请求字段为 `query`、`code`、`threshold`；
4. 响应成功时使用 `pruned_code`，失败时保留原 observation；
5. trajectory 的 user message 保存
   `origin_token_cnt/left_token_cnt/model_input_token_cnt`；
6. batch runner 暴露 `--pruner-url` 与 `--disable-pruner`。

不要在本仓库脚本里用正则或 `sed` 修改 site-packages；那会让实验无法
审计。应使用服务器已有的 pruning-capable mini-swe-agent 安装，或在其
自身仓库中完成上述小改动后重新安装。

如 base config 不在包内默认位置，可只指定路径：

```bash
MINI_SWE_BASE_CONFIG=/absolute/path/to/pruner.yaml \
bash scripts/run_server_experiments.sh preflight
```
