# 执行信号 + AST 骨架剪枝

这个任务针对 coding agent 常见工具输出使用确定性规则，不训练或加载任何模型。
它实现统一的 `tf_pruning.protocol.Pruner`，并且只依赖 Python 标准库。

## 支持的观测

`tool_type` 支持 `source`、`grep`、`traceback`、`diff`、`test_log`、
`tree` 和 `generic`。使用默认值 `auto` 时，剪枝器会先看 metadata 中的
`tool`、`tool_name`、`command`、`cmd`，再识别输出特征：

- source：保留 import、类/函数签名和报错行；
- grep：保留 `path:line:match` 命中及定义行；
- traceback：保留 traceback 头、frame、异常行及其窗口；
- diff：保留文件/hunk header 和 `+`/`-` 变化行；
- test_log：保留失败摘要、assert/error、`file:line` 及其窗口；
- tree：保留浅层骨架、query 命中节点及其祖先；
- generic：回退到结构声明、错误信号和 query 命中。

对于可解析的 Python source，会额外构建 AST：

1. import 与所有类/函数签名形成最小骨架；
2. query 或 recent context 命中的符号展开完整定义体；
3. 可选展开本地被调用者和调用者的一跳定义体；
4. 最终仍由 `PruningRequest.budget` 执行硬行数上限。

如果 Python 代码不完整、带命令输出前缀或存在语法错误，会自动回退到语言无关的
结构正则，不会导致请求失败。

## 输入与运行

JSONL 每行可直接写 request，也可写成 `{"request": {...}}`：

```json
{
  "request_id": "source-1",
  "text": "import json\n\ndef helper(x):\n    return x\n\ndef target(x):\n    return helper(x)",
  "query": "inspect target",
  "tool_type": "auto",
  "path": "src/service.py",
  "metadata": {"command": "sed -n '1,200p' src/service.py"},
  "budget": {
    "keep_ratio": 0.7,
    "min_lines": 1,
    "no_prune_below": 0,
    "context_window": 1
  }
}
```

从 `training_free_pruning` 目录运行：

```bash
python -m tasks.execution_ast.cli \
  --input requests.jsonl \
  --output results.jsonl \
  --config tasks/execution_ast/config.example.json
```

也可以从 Python 通过统一工厂创建：

```python
from tasks.execution_ast import build_pruner
from tf_pruning.protocol import PruningRequest

pruner = build_pruner({"expand_one_hop": True})
result = pruner.prune(PruningRequest(text=output, tool_type="auto"))
print(result.metadata["detected_tool_type"])
```

输出的 metadata 会记录实际识别类型、AST 是否解析成功、命中和一跳展开的符号、
强制保留行与扩张种子；`line_scores[*].reasons` 记录每条规则的来源，便于做工具
类型分层和规则消融。

## 测试

```bash
python -m pytest tests/test_execution_ast.py
```
