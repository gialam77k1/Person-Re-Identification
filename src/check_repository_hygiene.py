from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".onnx", ".pth", ".pt", ".npy", ".npz", ".pkl", ".joblib"}
FORBIDDEN_NAMES = {".env", ".env.mlops", "credentials.json", "service-account.json"}
TEXT_SUFFIXES = {
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def tracked_files(root: Path) -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    return [root / value.decode("utf-8") for value in output.split(b"\0") if value]


def secret_patterns() -> list[tuple[str, re.Pattern[str]]]:
    return [
        ("private key", re.compile("-----BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
        ("GitHub token", re.compile(r"gh[pousr]_[A-Za-z0-9_]{30,}")),
        ("AWS access key", re.compile(r"AKIA[0-9A-Z]{16}")),
        ("Slack token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,}")),
    ]


def inspect_files(files: list[Path], root: Path, max_bytes: int) -> list[str]:
    errors: list[str] = []
    patterns = secret_patterns()
    for path in files:
        relative = path.relative_to(root)
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > max_bytes:
            errors.append(f"Tracked file exceeds {max_bytes} bytes: {relative} ({size} bytes)")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"Model/data binary must not be tracked by Git: {relative}")
        if path.name.lower() in FORBIDDEN_NAMES:
            errors.append(f"Secret-bearing filename must not be tracked by Git: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES or size > 1024 * 1024:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in patterns:
            if pattern.search(text):
                errors.append(f"Potential {label} found in tracked file: {relative}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Reject large model files and common secrets in Git.")
    parser.add_argument("--max-bytes", type=int, default=5 * 1024 * 1024)
    args = parser.parse_args()
    errors = inspect_files(tracked_files(ROOT), ROOT, args.max_bytes)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("Tracked repository size, binary and secret checks: PASS")


if __name__ == "__main__":
    main()
