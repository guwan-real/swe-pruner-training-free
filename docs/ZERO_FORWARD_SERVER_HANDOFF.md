# B200 Server Handoff: Zero-Forward SWE-Bench

This document is for the local agent on the B200 server. Do not start another
vLLM, install another mini-swe-agent, or run the removed posterior-likelihood
scripts.

## Fixed resources

Repository:

```text
/home/yuantao/futao/swepruner_training_free_workspace/swe-pruner-training-free
```

Existing agent model:

```text
http://127.0.0.1:8015/v1
Qwen3.5-27B
```

Ports `8121`–`8124` are CPU pruning/recovery services. They never call port
`8015`; only mini-swe-agent calls Qwen.

## First-time setup

```bash
cd /home/yuantao/futao/swepruner_training_free_workspace/swe-pruner-training-free
git pull --ff-only
bash scripts/create_server_conda.sh
cp configs/zero_forward_server_profile.example.env zero_forward_server_profile.env
```

Edit the untracked profile:

```text
MINI_SWE_PYTHON=/absolute/path/to/existing/mini-swe-agent-venv/bin/python
```

The primary target is the Python environment for the official SWE-Pruner eval
fork at:

```text
downstream_eval/multi_turn/swebench/mini-swe-agent--with-pruning
```

Standard mini-swe-agent remains a secondary compatibility path. The
environment-creation script and launcher both remove an inherited uv/venv
before activating the project conda. The launcher remembers and invokes this
exact mini Python for agent runs.

`MINI_SWE_BASE_CONFIG` is optional. By default the launcher discovers
`extra/swebench.yaml` or `benchmarks/swebench.yaml` from that same mini
installation. Set it only when the server fork uses a custom base YAML.

## Mandatory preflight

```bash
bash scripts/run_zero_forward_swebench.sh preflight
```

It verifies:

1. project package import;
2. official fork `execute_action(self, action)`/`add_message` contract, or the
   secondary standard batch contract;
3. a supported mini SWE-Bench runner;
4. Docker daemon and disk space;
5. Qwen model discovery from `8015/v1/models`;
6. an ephemeral `/prune` request;
7. zero model forwards/tokens;
8. task evidence retention;
9. byte-for-byte raw recovery.

It does not send a pruning request to Qwen.

For the official fork, successful output includes:

```text
"mini_mode": "swe-pruner-single-v1"
"legacy_pruner_strategy": "empty-config-preserve-context-focus-question"
```

The zero-forward launcher intentionally does not pass `--disable-pruner`.
The generated shared YAML removes `agent.pruner`; the fork runner recreates
only an empty dictionary, which does not construct `PrunerClient`. This keeps
the existing `context_focus_question` prompt for both baseline and pruning
arms. Do not manually add `--pruner-url` or `--disable-pruner`.

Config discovery sets `MSWEA_SILENT_STARTUP=1` so the SWE-Pruner fork's startup
banner cannot pollute the captured YAML path.

## One-task smoke

```bash
bash scripts/run_zero_forward_swebench.sh smoke
bash scripts/run_zero_forward_swebench.sh status
bash scripts/run_zero_forward_swebench.sh results
```

Smoke runs the same Verified task twice:

- `baseline`: no hook;
- `adaptive_evidence`: tool-boundary CPU pruning.

Valid results require:

```text
pruner_model_forwards = 0
pruner_llm_tokens = 0
pruner_model_input_tokens = 0
pruning_errors = 0
```

Use the trajectory summary for agent counts:

```text
pruning_attempts    all tool outputs seen by the adapter
client_skips        outputs shorter than ZERO_FORWARD_MIN_CHARS
server_requests     actual POST /prune calls from the agent
server_pruned       runtime observations compacted by the service
server_skipped      runtime service cost-gate skips
recovery_guarded    full recovery echoes withheld from persistent history
```

`results` also reports each arm's prompt-token and wall-time percentage versus
the baseline. The CSV/JSON contains exact API-call, prompt-token, total-token,
wall-time and resolve-rate deltas. Comparisons are emitted only when both arms
have the same non-zero trajectory count.

Service `/metrics` also sees the startup preflight fixture. Its legacy
`requests/pruned/skipped` totals therefore include that probe; use
`runtime_requests/runtime_pruned/runtime_skipped` for agent traffic.

Keep `MAX_OUTPUT_CHARS=9000` unless the base mini YAML uses a different
long-observation limit. It ensures all selected evidence reaches Qwen instead
of being clipped a second time by mini-swe-agent.

Then grade:

```bash
bash scripts/run_zero_forward_swebench.sh grade
bash scripts/run_zero_forward_swebench.sh results
```

## Five experiment arms

```bash
bash scripts/run_zero_forward_swebench.sh launch
```

Default arms:

| Arm | Port | Added LLM forward |
|---|---:|---:|
| baseline | none | 0 |
| safe_rules | 8121 | 0 |
| intent_ir | 8122 | 0 |
| intent_structure | 8123 | 0 |
| adaptive_evidence | 8124 | 0 |

`PARALLEL_ARMS=1` is useful for quality comparison. Set it to `0` and choose a
new `RUN_TAG` for clean wall-time comparison because concurrent arms share
Qwen's scheduler.

`result` and `results` are equivalent. Stop only this run with:

```bash
bash scripts/run_zero_forward_swebench.sh stop
```

It does not stop vLLM or the official SWE-Pruner process.

## Raw recovery

The generated mini YAML adds:

```text
--add-host=host.docker.internal:host-gateway
```

A compact observation contains a `curl` command pointing to its arm's service.
It instructs the agent to save the output and inspect bounded ranges, never
`cat` the complete file. `ZERO_FORWARD_RECOVERY_MAX_CHARS` defaults to `3000`;
larger recovery output is replaced with a short receipt after the command
executes, while the saved file remains available. Recovery actions and guarded
echoes are counted separately in `summary.csv`. The raw store is under:

```text
<run-root>/raw/<method>/
```

Do not delete it before trajectories finish. Items expire according to
`RAW_TTL_HOURS`.
