# IR + execution/AST hybrid

This online, training-free method combines the two CPU rankers used in the
first coding-agent experiment:

- IR/BM25, identifier, path and recent-context relevance;
- command-output evidence, traceback/test/diff/grep rules and Python AST
  structure.

Raw scores from the two methods are not on the same scale, so the hybrid uses
deterministic rank-percentile fusion. Critical execution evidence is protected
before the remaining hard line budget is filled. The default weights are
static (`0.55 / 0.45`) and are not trained or calibrated.

```bash
python -m tf_pruning.cli prune \
  --method ir_ast_hybrid \
  --config tasks/ir_ast_hybrid/config.example.json \
  --request examples/requests/source_file.json
```

For the real mini-swe-agent experiment, use
`bash scripts/run_server_experiments.sh`; this task is served on its own local
HTTP port alongside the IR-only and AST-only arms.
