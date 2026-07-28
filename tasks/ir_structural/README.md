# IR + 结构锚点剪枝

这个任务用纯 Python 实现训练免费的行级剪枝器，不需要模型权重、GPU、向量库或额外
API 调用。它遵循统一的 `tf_pruning.protocol.Pruner` 接口，可以直接替换外部
skimmer。

## 算法

每一行的最终分数由以下固定信号相加：

- 对行及其相邻窗口计算 Okapi BM25；
- 计算 query/recent context 与行中标识符的重叠；
- 对出现在输出中的文件名和路径片段增加 path prior；
- 对最近上下文中的词和标识符增加 recent-context bonus；
- 给 import、类、函数和接口声明等结构锚点固定 bonus；
- 给报错行固定 bonus。

选择阶段以 `PruningRequest.budget` 作为硬行数预算。结构锚点优先保留，高分语义
命中与结构锚点作为窗口扩张种子；窗口大小由统一预算里的 `context_window`
控制。若锚点超过预算，仍会严格遵守预算，并按混合分数确定优先级。

## 输入协议

输入是 JSONL，每行可以直接是 request，也可以写成 `{"request": {...}}`：

```json
{
  "request_id": "sample-1",
  "text": "import os\n\ndef load_config(path):\n    return read(path)",
  "query": "where is load_config implemented",
  "tool_type": "source",
  "path": "src/config.py",
  "recent_context": ["read config failure"],
  "budget": {
    "keep_ratio": 0.5,
    "min_lines": 1,
    "no_prune_below": 0,
    "context_window": 1
  }
}
```

`PruningResult` 会返回保留行号、带省略标记的文本、每行分数及可解释的
`reasons`。默认会给保留行加原始行号，便于 agent 继续精确定位。

## 运行

从 `training_free_pruning` 目录运行：

```bash
python -m tasks.ir_structural.cli \
  --input requests.jsonl \
  --output results.jsonl \
  --config tasks/ir_structural/config.example.json
```

也支持管道输入输出：

```bash
printf '%s\n' '{"text":"def target():\n    return 1","query":"target"}' |
  python -m tasks.ir_structural.cli --input - --output -
```

Python 中的统一工厂入口：

```python
from tasks.ir_structural import build_pruner
from tf_pruning.protocol import PruningRequest

pruner = build_pruner({"weights": {"identifier": 3.0}})
result = pruner.prune(PruningRequest(text=source, query="target"))
```

配置字段见 `config.example.json`。`weights` 内的名字会自动映射到对应
`*_weight` 参数；错误的字段会立即报错，避免实验配置静默失效。

## 测试

```bash
python -m pytest tests/test_ir_structural.py
```
