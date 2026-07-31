# mini-swe-agent Adapter Contract

## Supported installations

The primary supported installation is the official SWE-Pruner evaluation fork:

```text
downstream_eval/multi_turn/swebench/mini-swe-agent--with-pruning
```

Runtime capability detection supports:

1. standard mini-swe-agent exposing
   `DefaultAgent.execute_actions(self, message)` plus `add_messages`;
2. the official SWE-Pruner eval fork exposing
   `DefaultAgent.execute_action(self, action)` plus `add_message`.

The launcher uses the existing mini Python through `MINI_SWE_PYTHON`. It does
not install or copy mini-swe-agent into the project conda environment.
Package version strings are not used for compatibility because both different
implementations can report version `1.16.0`. Preflight prints the detected mode
as `upstream-batch-v2` or `swe-pruner-single-v1`.

## Hook boundary

For standard mini, the adapter owns the existing batch tool boundary:

```python
output = agent.env.execute(action)
output["output"] = zero_forward_client.prune(output["output"])
messages = agent.model.format_observation_messages(message, outputs, ...)
agent.add_messages(*messages)
```

For the SWE-Pruner fork, the adapter wraps and calls the original single-action
method instead of copying it:

```text
original execute_action(action)
  -> env.execute(action["action"])
  -> timeout and has_finished checks
  -> legacy _apply_pruner (no-op because pruner_client is None)
  -> zero-forward CPU pruning
  -> output["pruned_stats"] for trajectory persistence
```

This preserves the fork's timeout, submission and future execution semantics.
No model method is patched. Baseline uses the same runner and shared YAML but
sets `ZERO_FORWARD_ALLOW_BASELINE=1` without installing the hook.

The adapter refuses to run when the fork's `agent.pruner` is still active,
preventing accidental double pruning. The config adapter removes that section
and adds Docker host-gateway access for raw recovery.

Do not pass `--disable-pruner` through this wrapper for the official fork. Its
runner uses that flag to replace the pruning-aware prompts with templates that
do not request `context_focus_question`. With the flag absent, the runner
restores only an empty `pruner: {}`; this value is false in `DefaultAgent`, so
no `PrunerClient` is constructed and the focus prompt remains. Baseline and
pruning arms therefore use identical prompts and neither activates the legacy
HTTP pruner.

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

Standard mini stores the environment stats at:

```text
extra.zero_forward_pruning
```

The SWE-Pruner fork additionally receives the same mapping at
`output.pruned_stats`; its existing `get_observation` method copies that mapping
onto the user trajectory message. The model-facing observation template still
renders only `output["output"]`.

Preflight checks the exact hook signature and supports both historical
`minisweagent.run.extra.swebench` and current
`minisweagent.run.benchmarks.swebench` module locations.
