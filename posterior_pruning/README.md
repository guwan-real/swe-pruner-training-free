# Isolated post-action posterior pruning

This directory implements the training-free posterior experiments. It does not
import `tf_pruning`, register methods in `tf_pruning.registry`, or change the
legacy `integrations/http_server.py`.

The defining timing is:

1. mini-swe-agent receives a complete tool observation;
2. Qwen generates the real next action from that complete observation;
3. this package evaluates counterfactual compacted observations by scoring that
   fixed action with the same frozen vLLM;
4. only the stored observation is compacted, so the change can affect future
   turns but cannot retroactively change the current action.

The service contract is `POST /prune-post-action`, not the legacy `POST
/prune`. See `docs/POSTERIOR_PRUNING_DESIGN.md` for the objective and methods,
`docs/POSTERIOR_MINI_SWE_AGENT_ADAPTER.md` for the exact hook, and
`docs/POSTERIOR_SERVER_HANDOFF.md` for server commands.

Subdirectories:

- `methods/single_verify`: full score plus one candidate score;
- `methods/budget_search`: bounded search for the smallest safe budget;
- `methods/greedy_blocks`: posterior-verified greedy block deletion;
- `methods/block_influence`: bounded leave-one-block-out oracle;
- `mini_adapter`: runtime wrapper around the installed mini-swe-agent;
- `agent_eval`: trajectory and official grader aggregation.

No method creates a checkpoint or updates model weights.
