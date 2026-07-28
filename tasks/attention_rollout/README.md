# Attention rollout and heavy-hitter pruning

This task reads attention already produced by an open-weight agent and ranks
tool-response lines without training a scorer. It supports direct decode
attention mass (the inexpensive default) and classic residual attention
rollout over full square attention matrices.

## Input contract

Use the shared `PruningRequest` JSONL schema. Runtime arrays can be ordinary
lists under `request.metadata`, or stored in an NPZ:

```json
{
  "text": "def load():\n    return cache\nerror: stale value",
  "budget": {
    "keep_ratio": 0.5,
    "no_prune_below": 0,
    "context_window": 1
  },
  "metadata": {
    "attention_path": "/data/sample-17.attention.npz",
    "sink_token_indices": [0]
  }
}
```

For `attention_mass`, the default NPZ contract is:

- `attention`: `[layers, heads, decode_steps, response_tokens]`
- `token_to_line`: one line id per response token

Lower-rank forms `[tokens]`, `[steps, tokens]`, and
`[heads, steps, tokens]` are inferred. A different order can be declared with
`attention_layout`, such as `"steps,layers,heads,tokens"`. A leading batch axis
is also supported.

For `rollout`, use square attention matrices:

- `attention`: `[layers, heads, sequence, sequence]`
- `token_to_line`: one line id per key token; use `-1` for prompt/decode tokens
  that do not belong to the response
- `metadata.decode_token_indices`: query rows whose rollout should score the
  response

The inferred rollout layout is `layers,heads,queries,keys`. Direct arrays use
the same names. NPZ loading imports NumPy lazily, so list-only use has no NumPy
dependency.

## Aggregation and selection

Layer, head, and decode-step selectors accept `all`, `first`, `first-N`,
`last`, `last-N`, an integer, or an index list. Each axis can aggregate with
`mean`, `max`, or `sum`; decode steps additionally support recency-weighted
aggregation via `weighted` and `step_decay`.

`sink_first_tokens`, configured sink indices, and per-request
`metadata.sink_token_indices` zero attention-sink contributions before token
scores are pooled to lines. Scores are then normalized. Code structure lines
receive `structure_floor`, while neighbors of the strongest lines receive
`local_floor`; these are floors, not unconditional keeps, so the hard budget is
never exceeded.

Selection always uses `tf_pruning.selection.select_line_numbers`:

- `hard_budget`: keep exactly the shared budget target.
- `top_p`: take the smallest attention nucleus reaching `top_p`, capped by the
  shared hard budget.
- `hybrid`: use Top-P lines as context-expansion seeds and fill the shared hard
  budget by rank.

## Run

From `training_free_pruning`:

```bash
python -m tasks.attention_rollout.cli \
  requests.jsonl \
  --config tasks/attention_rollout/config.example.json \
  --output attention-results.jsonl
```

For rollout, override the example config with:

```json
{
  "pruner": {
    "method": "rollout",
    "attention_layout": "layers,heads,queries,keys",
    "layers": "last-4",
    "decode_steps": "last-3"
  }
}
```

Python callers can use `tasks.attention_rollout.build_pruner(config)`. Result
metadata records the actual tensor layout, selected layer/head/step indices,
sink mask, Top-P nucleus, and structural/local seeds for reproducible
ablations.
