# Influence-function-style pruning oracle

This task estimates the importance of a tool-response block by measuring how
much deleting it harms the log likelihood of the *recorded next agent action*.
It updates no weights. Because it needs many causal-LM forwards, it is an
offline, small-sample oracle—not an online default pruner.

## Strategies

- `leave_one_out`: score the full observation once, remove each code-aware
  block independently, and assign
  `full_log_likelihood - ablated_log_likelihood` to that block. The shared hard
  line-budget selector keeps the highest-harm lines.
- `hierarchical_greedy`: repeatedly evaluate every legal block deletion in the
  current observation and remove the least harmful block. When no whole block
  fits the remaining exact budget, refine retained blocks to individual lines
  and continue. Candidate batches are evaluated completely to avoid
  budget-order bias.

The objective is deliberately pluggable. The default
`NextActionObjective` reads `metadata.next_action`, builds a prefix from the
query, path, recent context, and a line-numbered omission skeleton, then asks a
scorer for the reference action log likelihood. Alternative objectives and
scorers can be injected through Python.

## Environment and offline model

```bash
cd /path/to/training_free_pruning
python -m pip install -e '.[model]'
```

The Hugging Face scorer is lazy: imports and weight loading occur only on the
first scored request. Tokenizer and model always use `local_files_only=True`,
and false is rejected. Copy a causal coder LM to a local/server directory and
set `model_path` to it. No model download is performed by this task.

By default log likelihood is the sum across all tokens in the recorded action.
Set scorer `normalize=true` to use mean token log likelihood. Since every
ablation within a request has the same target action, summed likelihood is the
natural default.

## Replay input

Each JSONL row follows `tf_pruning.protocol.PruningRequest`. The recorded next
action is mandatory:

```json
{"request_id":"trace-7","text":"...long tool output...","query":"fix parse regression","tool_type":"read","path":"src/parser.py","recent_context":["test_parse_nested failed"],"budget":{"keep_ratio":0.5,"no_prune_below":0},"metadata":{"next_action":"{\"path\":\"src/parser.py\",\"line_end\":220}"}}
```

Run leave-one-out with a copied config:

```bash
python -m tasks.influence_oracle.cli \
  --config tasks/influence_oracle/config.example.json \
  --input data/replay_requests.jsonl \
  --output runs/influence_loo/results.jsonl
```

For hierarchical greedy, change `pruner.strategy`. Start with 50–100 replay
examples, coarse blocks, and a conservative context length. The result metadata
records exact scorer evaluations, full-action likelihood, and the explicit
`small-sample-offline` scope. `max_initial_blocks` and `max_evaluations` fail
fast rather than silently return a biased partial oracle.

## Programmatic extension

```python
from tasks.influence_oracle import InfluenceOraclePruner

pruner = InfluenceOraclePruner(
    scorer=my_log_likelihood_scorer,
    objective=my_preservation_objective,
)
result = pruner.prune(request)
```

A scorer implements
`log_likelihood(context, continuation) -> float`. An objective implements
`target(request) -> str` and `prompt(request, observation) -> str`.
`build_pruner()` returns the default objective/pruner without loading a model;
actual compression requires a configured `model_path` or an injected scorer.
