# 输入与输出格式

## 统一请求

单条 JSON：

```json
{
  "request_id": "task-001-step-04",
  "text": "tool response 原文",
  "query": "Goal Hint、用户问题或自动提取的意图",
  "tool_type": "source",
  "path": "src/client.py",
  "recent_context": [
    "最近提到的报错、符号或路径"
  ],
  "budget": {
    "keep_ratio": 0.5,
    "min_lines": 8,
    "max_lines": null,
    "no_prune_below": 20,
    "context_window": 1
  },
  "metadata": {}
}
```

`tool_type` 可设为 `auto`，也可显式使用 `source`、`grep`、`traceback`、`test_log`、`diff`、`tree` 或 `generic`。

所有行号都是 **从 1 开始**，与用户看到的 tool response 行号一致。

## 模型内部信号

不同方法会在 `metadata` 中读取不同输入：

- hidden-state 方法：token hidden states、token-to-line 映射、anchor states，或对应 `.npz` 路径；
- attention 方法：attention tensor、token-to-line 映射，或 `.npz` 路径；
- PPL/influence 方法：本地模型由 method config 指定，request 中提供 action/query。

数组键名、shape 与 `.npz` 示例以对应任务目录的 README 为准。数组不应直接写进大规模 JSONL，推荐保存为只读 `.npz`，在 metadata 中放相对或绝对路径。

## Replay 标签

```json
{
  "request": {
    "request_id": "trace-1",
    "text": "...",
    "query": "fix timeout failure",
    "tool_type": "traceback"
  },
  "gold_line_numbers": [3, 4, 5, 6],
  "required_line_numbers": [4, 5, 6]
}
```

- `gold_line_numbers`：用于 line precision/recall/F1 的完整相关行；
- `required_line_numbers`：一旦漏掉就可能破坏任务的关键行，用于 critical miss；
- 两类标签都可缺省，此时仍会统计压缩率、耗时与 forward 次数。

## 统一结果

```json
{
  "method": "ir_structural",
  "request_id": "task-001-step-04",
  "original_line_count": 100,
  "kept_line_count": 45,
  "retention_ratio": 0.45,
  "kept_line_numbers": [1, 8, 9],
  "pruned_text": "... <7 lines pruned> ...",
  "line_scores": [
    {
      "line_no": 1,
      "score": 2.1,
      "reasons": ["structure_anchor"]
    }
  ],
  "latency_ms": 2.7,
  "metadata": {
    "model_forward_count": 0
  }
}
```

骨架文本中的省略范围必须显式出现，避免 agent 把不相邻的代码误认为连续代码。
