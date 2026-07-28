# Conditional PPL / surprisal pruning

This task replaces a trained observation skimmer with a local causal LM used
only as a scoring function. It is training-free: no weights or task-specific
head are updated.

## Algorithm

1. Split the tool response with `tf_pruning.text.code_aware_blocks`; long and
   uncovered regions are chunked so every line has a coarse candidate.
2. Score every block by conditional surprisal given the goal, recent context,
   tool type, and path.
3. Refine only the highest-surprisal blocks (plus blocks containing protected
   anchors) with line-level conditional surprisal.
4. Fuse block and line scores, then use the shared
   `tf_pruning.selection.select_line_numbers` hard budget.
5. Protect structure declarations, errors, and optional
   `metadata.anchor_lines`, and render the shared line-numbered skeleton.

The default implements the mixed objective from the research plan:
`coarse_first_token_only=false` scores mean block NLL, while
`first_token_only=true` uses first-token surprisal during fine line ranking.
Either switch can be changed independently for ablations. Higher surprisal is
treated as more informative. The method is intended for offline replay or
moderate-size local scorers; always report scorer calls and wall time.

## Environment

Use the same Python 3.11 environment as the project:

```bash
cd /path/to/training_free_pruning
python -m pip install -e '.[model]'
```

The scorer performs lazy imports and lazy model loading. Both tokenizer and
model are loaded with `local_files_only=True`; setting it to false is rejected.
Copy an already-downloaded Hugging Face causal LM to the server and point
`model_path` at that local directory. Importing this package never accesses the
network.

## Input and run

Input is the shared request JSONL schema:

```json
{"request_id":"ex-1","text":"...","query":"find parser failure","tool_type":"read","path":"src/parser.py","recent_context":["tests fail in parse_value"],"budget":{"keep_ratio":0.5,"no_prune_below":20},"metadata":{"anchor_lines":[17]}}
```

Copy and edit `config.example.json`, then run:

```bash
python -m tasks.conditional_ppl.cli \
  --config tasks/conditional_ppl/config.example.json \
  --input data/requests.jsonl \
  --output runs/conditional_ppl/results.jsonl
```

The output is the shared `PruningResult` JSONL. Its metadata records block
counts, refined line counts, scorer calls, protected anchors, and scoring mode.

Programmatic use supports a mock or alternative scorer:

```python
from tasks.conditional_ppl import ConditionalPPLPruner

pruner = ConditionalPPLPruner(scorer=my_conditional_surprisal_scorer)
result = pruner.prune(request)
```

The scorer must implement
`score(context, continuation, first_token_only=False) -> float`, returning
surprisal in nats. `build_pruner()` returns a default pruner without loading a
model; pruning a response that actually needs compression requires either an
injected scorer or a configured `model_path`.
