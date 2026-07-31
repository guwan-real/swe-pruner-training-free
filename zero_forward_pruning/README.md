# Zero-Forward Pruning

This package is an isolated online implementation for coding-agent tool output.
It does not import `tf_pruning` or the removed posterior-likelihood package.

Hard runtime invariants:

- no trained checkpoint or pruning head;
- no vLLM/OpenAI request from the pruner;
- `model_forward_count == 0`;
- `model_input_token_cnt == 0`;
- `llm_token_count == 0`;
- fail-open when parsing, recovery storage, or cost gates fail.

The primary agent integration target is SWE-Pruner's official
`downstream_eval/multi_turn/swebench/mini-swe-agent--with-pruning` fork. Its
single-action hook is detected from method signatures, not the unreliable
package version. Standard mini's batch action API remains supported.

## Methods

- `methods/safe_rules`: hard errors, trace frames, source locations and source
  skeletons;
- `methods/intent_ir`: sparse query/identifier retrieval;
- `methods/intent_structure`: intent retrieval plus neighbouring blocks;
- `methods/adaptive_evidence`: coverage-driven recommended method with no
  keep-ratio search.

All methods use the same official-compatible endpoint:

```json
{
  "query": "Where is resolve_model validated?",
  "code": "... raw tool output ...",
  "threshold": 0.5,
  "command": "sed -n '1,400p' model.py",
  "path": "model.py",
  "task": "Fix model validation"
}
```

The first three fields are compatible with the SWE-Pruner `/prune` contract.
The remaining fields are optional zero-cost intent signals from the existing
agent action.

Run a service:

```bash
zero-forward-prune-serve \
  --method adaptive_evidence \
  --port 8124 \
  --raw-store /tmp/zero-forward-raw
```

Probe pruning and byte-exact recovery:

```bash
zero-forward-prune-preflight --url http://127.0.0.1:8124
```

The service exposes `/health`, `/metrics`, `/prune`, and an unguessable
`/raw/<id>` recovery route. Stored observations expire according to
`--raw-ttl-hours`.
