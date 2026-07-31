# Zero-Forward Coding-Agent Pruning Design

## 1. Non-negotiable objective

The online pruner optimizes complete trajectory cost, not deletion likelihood:

```text
minimize agent wall time + agent prompt/completion tokens
subject to task quality
```

The following are runtime invariants rather than experimental targets:

```text
trained pruning parameters = 0
additional pruner LLM calls = 0
additional pruner model forwards = 0
additional pruner LLM tokens = 0
```

The removed implementation rescored a fixed action under several deleted
contexts. That was useful as an offline influence oracle but cost two to eight
full-context forwards per prunable turn. It is not part of this package or the
server launcher.

## 2. Timing

The primary adapter wraps the official SWE-Pruner eval fork's
`DefaultAgent.execute_action(action)`. Standard mini's
`DefaultAgent.execute_actions(message)` is retained as a secondary capability
path:

```text
normal agent inference
  -> generated bash/tool action
  -> environment.execute(action)
  -> raw output
  -> original submission/timeout checks
  -> zero-forward CPU pruner
  -> existing observation formatter
  -> compact observation enters Qwen
```

Therefore the raw long output is never sent to Qwen for that turn. The pruned
form also becomes the stable history for later turns.

The intent signal is free: task text, the already generated command,
`context_focus_question` when present, paths, identifiers and recent messages.
No goal-hint generation request is added.

For the official fork, the generated shared config removes the non-empty
legacy `agent.pruner` section but does not use `--disable-pruner`. The runner
therefore creates a false empty pruner config without constructing
`PrunerClient`, while retaining the prompts that produce
`context_focus_question`. A runtime guard rejects any non-null legacy client
before executing the action.

## 3. Evidence pipeline

The service classifies each observation as source, diff, traceback, test log,
search output, tree output or generic text. It creates bounded blocks and
separates hard from soft evidence.

Hard evidence includes:

- errors, assertions and failure summaries;
- traceback frames and source locations;
- diff hunks;
- source imports and declarations;
- first/last provenance lines.

Soft evidence is ranked by:

- exact identifier overlap;
- BM25-style sparse retrieval;
- paths and source locations;
- tool-type evidence.

Ranks are combined with reciprocal-rank fusion. There is no fitted classifier
or learned weighting vector. Selected source/log blocks are expanded to
neighbouring blocks to avoid returning isolated syntax.

## 4. Four ablations

| Method | Query-aware | Structural expansion | Runtime ratio search |
|---|---:|---:|---:|
| `safe_rules` | no | limited | no |
| `intent_ir` | yes | no | no; one contract budget |
| `intent_structure` | yes | yes | no; one contract budget |
| `adaptive_evidence` | yes | yes | no |

`adaptive_evidence` is the recommended method. It ignores `threshold` for
selection, retains hard evidence, covers query terms found in the observation,
then adds bounded exact-identifier evidence. The threshold remains in the HTTP
request only for compatibility with existing SWE-Pruner consumers.

## 5. Cost gates

The pruner returns the complete original output when:

- the output is below `MIN_INPUT_TOKENS`;
- the output is a diff;
- no safe evidence reduction exists;
- rendered savings are below `MIN_SAVINGS_TOKENS`;
- retention is above `MAX_RETENTION_RATIO`;
- CPU work exceeds `MAX_CPU_MS`;
- raw recovery storage fails;
- parsing or selection raises an exception.

Omission markers and the recovery banner are included in the reported retained
token estimate, so marker overhead cannot create fake savings.

Selected evidence is additionally fit below `MAX_OUTPUT_CHARS` (default 9000).
This stays under standard mini-swe-agent's 10,000-character observation display
limit, preventing a relevant block from being selected by the pruner and then
discarded by mini's own head/tail fallback.

## 6. Reversibility

Before returning a compact view, the service atomically stores the exact UTF-8
raw output under a cryptographically random ID. The compact view includes:

```text
curl -fsS 'http://host.docker.internal:PORT/raw/RANDOM_ID' \
  -o '/tmp/zero-forward-RANDOM_ID.txt'
```

The generated mini config adds Docker's
`host.docker.internal:host-gateway` mapping. Source files can also be reread
normally. Stored outputs expire after `RAW_TTL_HOURS`; IDs are path-validated
and directory traversal is rejected. Saving to a file avoids mini-swe-agent's
own long-observation display limit; the agent can then inspect precise chunks
with `sed`, `rg`, `head`, or `tail`.

If storage is unavailable, the request fails open instead of returning an
irreversible truncation.

The mini adapter recognizes an explicit `curl`/`wget` request to `/raw/<id>`
and passes that single recovered output through unchanged, preventing recursive
re-pruning.

## 7. Metrics and acceptance

Every trajectory summary reports:

- agent calls and prompt/completion tokens;
- pruned/skipped/error counts;
- pruner CPU latency;
- observation retention;
- recovery actions;
- pruner model forwards, model-input tokens and LLM tokens;
- wall time and official SWE-Bench resolve rate.

A run is invalid if any pruning arm reports a non-zero pruner model-forward or
LLM-token count. Practical acceptance should require fewer agent prompt tokens,
lower end-to-end wall time, no meaningful increase in agent rounds/recovery,
and comparable official resolve rate.
