from __future__ import annotations

import argparse
import inspect
import json

from posterior_history_pruning.mini_adapter.hook import detect_installed_mini_mode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate posterior-history mini-swe-agent contract"
    )
    parser.parse_args(argv)
    mode = detect_installed_mini_mode()
    try:
        from minisweagent.run.extra import swebench
    except ImportError:
        try:
            from minisweagent.run.benchmarks import swebench
        except ImportError as exc:
            raise SystemExit("cannot import a supported mini-swe-agent SWE-Bench runner") from exc
    try:
        parameters = tuple(inspect.signature(swebench.main).parameters)
    except (TypeError, ValueError) as exc:
        raise SystemExit("cannot inspect SWE-Pruner eval runner main()") from exc
    missing = sorted({"pruner_url", "disable_pruner"}.difference(parameters))
    if missing:
        raise SystemExit(
            "SWE-Pruner eval runner contract changed; missing parameters: " + ", ".join(missing)
        )
    print(
        json.dumps(
            {
                "status": "passed",
                "mini_mode": mode,
                "runner": swebench.__name__,
                "query_boundary": "DefaultAgent.query -> model.query(messages)",
                "model_forward_count": 0,
                "llm_token_count": 0,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
