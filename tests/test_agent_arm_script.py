from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_one_agent_arm.sh"


def test_agent_arm_forwards_arguments_without_splitting(tmp_path: Path) -> None:
    arm_dir = tmp_path / "arm output"
    recorder = tmp_path / "record args.py"
    recorded = tmp_path / "recorded.json"
    recorder.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "from pathlib import Path",
                f"Path({str(recorded)!r}).write_text(json.dumps(sys.argv[1:]))",
            ]
        ),
        encoding="utf-8",
    )
    expected = [
        "swebench",
        "--filter",
        "value with spaces",
        "--config",
        "/tmp/config with spaces.yaml",
    ]

    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            str(arm_dir),
            sys.executable,
            str(recorder),
            *expected,
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(recorded.read_text(encoding="utf-8")) == expected
    assert (arm_dir / "exit_code").read_text(encoding="utf-8").strip() == "0"
    assert (arm_dir / "started_at").is_file()
    assert (arm_dir / "ended_at").is_file()
