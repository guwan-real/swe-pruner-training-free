# mini-swe-agent Adapter Contract

## Supported installations

The adapter supports:

1. standard mini-swe-agent exposing
   `DefaultAgent.execute_actions(self, message)`;
2. the SWE-Pruner mini fork when its legacy `agent.pruner` is disabled.

The launcher uses the existing mini Python through `MINI_SWE_PYTHON`. It does
not install or copy mini-swe-agent into the project conda environment.

## Hook boundary

Only `DefaultAgent.execute_actions` is replaced inside the runner process. The
adapter preserves the original order:

```python
output = agent.env.execute(action)
output["output"] = zero_forward_client.prune(output["output"])
messages = agent.model.format_observation_messages(message, outputs, ...)
agent.add_messages(*messages)
```

No model method is patched. Baseline uses the same runner and shared YAML but
sets `ZERO_FORWARD_ALLOW_BASELINE=1` without installing the hook.

The adapter refuses to run when the fork's `agent.pruner` is still active,
preventing accidental double pruning. The config adapter removes that section
and adds Docker host-gateway access for raw recovery.

An explicit `curl`/`wget` action targeting `/raw/<id>` bypasses pruning once.
Without this guard, the recovered long observation would immediately be
compacted again and recovery could loop.

## Request construction

For each action it sends:

- `code`: exact `output["output"]`;
- `query`: `context_focus_question`, otherwise the existing shell command;
- `command`: existing shell command;
- `path`: first path extracted from the command;
- `task`: SWE-Bench problem statement;
- `recent_context`: a bounded suffix of recent messages;
- return code and instance/call/action identifiers as metadata.

These are already available strings. Constructing the request performs no
tokenization or LLM inference.

## Trajectory fields

The environment output receives:

```text
extra.zero_forward_pruning
```

with method, status, token estimates, CPU latency, recovery information and the
three zero-valued LLM-cost counters. mini-swe-agent's observation formatter
copies this into trajectory message metadata without adding it to the model
content.

Preflight checks the exact hook signature and supports both historical
`minisweagent.run.extra.swebench` and current
`minisweagent.run.benchmarks.swebench` module locations.
