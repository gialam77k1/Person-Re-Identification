from __future__ import annotations

import argparse
import re
from pathlib import Path


_ROOT_CAUSE = re.compile(
    r"(?:AssertionError|CUDA out of memory|FileNotFoundError|KeyError|MemoryError|"
    r"OSError|PermissionError|RuntimeError|TypeError|ValueError):"
)
_WRAPPER_NOISE = (
    "ERROR conda.cli.main_run",
    "Error: Process completed with exit code",
)


def summarize_failure(text: str, *, limit: int = 900) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if _ROOT_CAUSE.search(line) and not line.startswith(_WRAPPER_NOISE):
            return line[-limit:]
    for line in reversed(lines):
        if not line.startswith(_WRAPPER_NOISE) and not line.startswith("Traceback"):
            return line[-limit:]
    return "Pipeline command failed without a captured error message"


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract a useful root cause from a pipeline log.")
    parser.add_argument("--log", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.log)
    summary = summarize_failure(source.read_text(encoding="utf-8", errors="replace"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(summary + "\n", encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
