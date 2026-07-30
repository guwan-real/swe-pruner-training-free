from __future__ import annotations

import inspect
import logging
from typing import Any, Mapping

from posterior_pruning.candidates import estimate_tokens
from posterior_pruning.mini_adapter.client import PosteriorClient, PosteriorClientConfig

LOGGER = logging.getLogger("posterior_pruning.mini_adapter")
_INSTALLED = False
_ORIGINAL_QUERY: Any = None


def assert_mini_compatible(default_agent_class: type) -> None:
    query = getattr(default_agent_class, "query", None)
    add_message = getattr(default_agent_class, "add_message", None)
    if not callable(query) or not callable(add_message):
        raise RuntimeError("mini-swe-agent DefaultAgent must expose query() and add_message()")
    signature = inspect.signature(query)
    if len(signature.parameters) != 1:
        raise RuntimeError(
            "mini-swe-agent DefaultAgent.query signature changed; expected query(self)"
        )


def _compact_stats(result: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "method",
        "status",
        "original_line_count",
        "kept_line_count",
        "original_estimated_tokens",
        "kept_estimated_tokens",
        "retention_ratio",
        "full_action_mean_logprob",
        "selected_action_mean_logprob",
        "mean_logprob_drop",
        "action_token_count",
        "model_forward_count",
        "scoring_prompt_tokens",
        "candidates_evaluated",
        "latency_ms",
        "error",
        "diagnostics",
    )
    return {field: result.get(field) for field in fields if field in result}


def _observation_index(messages: list[dict[str, Any]]) -> int | None:
    if len(messages) < 4:
        return None
    index = len(messages) - 2
    if messages[index].get("role") != "user":
        return None
    if messages[index - 1].get("role") != "assistant":
        return None
    if messages[-1].get("role") != "assistant":
        return None
    return index


def apply_after_query(agent: Any, response: Mapping[str, Any], client: PosteriorClient) -> None:
    messages = getattr(agent, "messages", None)
    if not isinstance(messages, list):
        raise RuntimeError("mini-swe-agent agent.messages must be a list")
    index = _observation_index(messages)
    if index is None:
        return
    observation = messages[index].get("content")
    next_action = response.get("content")
    if not isinstance(observation, str) or not isinstance(next_action, str) or not next_action:
        return
    history = [
        {
            "role": str(message.get("role", "")),
            "content": str(message.get("content", "")),
        }
        for message in messages[:-1]
    ]
    query = ""
    extra_vars = getattr(agent, "extra_template_vars", {})
    if isinstance(extra_vars, Mapping):
        task = extra_vars.get("task")
        if isinstance(task, str):
            query = task
    request_id = None
    instance_id = getattr(agent, "instance_id", None)
    model = getattr(agent, "model", None)
    call_count = getattr(model, "n_calls", None)
    if instance_id:
        request_id = f"{instance_id}:turn-{call_count or 0}"
    try:
        result = client.prune(
            messages=history,
            observation_index=index,
            next_action=next_action,
            query=query,
            request_id=request_id,
            metadata={"adapter": "mini-swe-agent-post-action-v1"},
        )
        pruned = result["pruned_response"]
        messages[index]["content"] = pruned
        messages[index]["posterior_pruned_stats"] = _compact_stats(result)
    except Exception as exc:
        LOGGER.warning("posterior pruning failed open: %s", exc)
        token_count = estimate_tokens(observation)
        line_count = len(observation.splitlines())
        messages[index]["content"] = observation
        messages[index]["posterior_pruned_stats"] = {
            "method": "client",
            "status": "client_error",
            "original_line_count": line_count,
            "kept_line_count": line_count,
            "original_estimated_tokens": token_count,
            "kept_estimated_tokens": token_count,
            "retention_ratio": 1.0,
            "model_forward_count": 0,
            "candidates_evaluated": 0,
            "error": str(exc),
        }


def install_hook(config: PosteriorClientConfig | None = None) -> bool:
    """Patch only ``DefaultAgent.query`` in the current mini process.

    The original query runs first and therefore generates the next action from
    the full observation.  The adapter then compacts the previous observation
    in ``agent.messages`` before any later model turn.
    """

    global _INSTALLED, _ORIGINAL_QUERY
    if _INSTALLED:
        return True
    config = config if config is not None else PosteriorClientConfig.from_env()
    if config is None:
        return False
    try:
        from minisweagent.agents.default import DefaultAgent
    except ImportError as exc:
        raise RuntimeError(
            "minisweagent is not importable; run the adapter with the mini-swe-agent Python"
        ) from exc
    assert_mini_compatible(DefaultAgent)
    client = PosteriorClient(config)
    original_query = DefaultAgent.query

    def query_with_posterior(self: Any) -> dict[str, Any]:
        if getattr(self, "pruner_client", None) is not None:
            raise RuntimeError(
                "legacy pre-action pruner is enabled together with posterior pruning; "
                "disable agent.pruner so the next action sees the full observation"
            )
        response = original_query(self)
        apply_after_query(self, response, client)
        return response

    _ORIGINAL_QUERY = original_query
    DefaultAgent.query = query_with_posterior
    _INSTALLED = True
    LOGGER.info(
        "installed post-action mini-swe-agent hook: %s keep_ratio=%.3f",
        config.endpoint,
        config.keep_ratio,
    )
    return True
