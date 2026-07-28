# Hidden-state anchor similarity

This task replaces a learned SWE-Pruner Pro classification head with a
deterministic cosine readout over hidden states already produced by the frozen
agent backbone. It never fits or loads a pruning head.

## Input contract

The executable reads the repository-wide `PruningRequest` JSONL format. Put
runtime representations in `request.metadata`, either as ordinary JSON arrays
or as an NPZ path:

```json
{
  "text": "line one\nline two\nline three",
  "query": "find the failed assertion",
  "budget": {
    "keep_ratio": 0.5,
    "no_prune_below": 0,
    "context_window": 1
  },
  "metadata": {
    "hidden_states_path": "/data/sample-17.hidden.npz"
  }
}
```

The NPZ must contain:

- `hidden_states`: `[tokens, hidden]` or `[layers, tokens, hidden]`
- `token_to_line`: one line id per response token
- any of `query_anchor`, `tool_anchor`, `error_anchor`, `decode_anchor`
  as `[hidden]`, `[tokens, hidden]`, or `[layers, tokens, hidden]`

The same names may be supplied directly under `metadata`. Anchor arrays may
also be grouped under `metadata.anchors`, and anchors may point into the
response states through `metadata.anchor_token_indices`, for example
`{"query": [4, 5], "decode": [31]}`. Anchor token indices are zero-based by
default. The line map accepts zero- or one-based ids (`line_map_base` can make
the choice explicit); negative ids are ignored as special/non-response tokens.

NPZ support imports NumPy only when a path is actually used. Direct list inputs
run on the Python standard library alone.

## Pooling and scoring

- `mean`: mean of a line's token vectors from the last layer.
- `max`: coordinate-wise maximum over its last-layer token vectors.
- `last`: the line's final token vector from the last layer.
- `last-4`: mean of its token vectors across the last four available layers.

Each line is compared with the available query, tool-call, error, and
first-decode anchors. Cosines are fused using `anchor_weights`, renormalized
over anchors that are actually present. If no explicit error anchor is
available, error/traceback lines can form one automatically. Ranking then goes
through `tf_pruning.selection.select_line_numbers`, so the shared hard line
budget and context-window behavior remain authoritative.

## Run

From `training_free_pruning`:

```bash
python -m tasks.hidden_state_similarity.cli \
  requests.jsonl \
  --config tasks/hidden_state_similarity/config.example.json \
  --output hidden-results.jsonl
```

For Python integration:

```python
from tasks.hidden_state_similarity import build_pruner

pruner = build_pruner({"pooling": "last-4"})
result = pruner.prune(request)
```

The result records line-level scores and per-anchor cosines for ablations, plus
the actual anchor set and tensor dimensions in result metadata.
