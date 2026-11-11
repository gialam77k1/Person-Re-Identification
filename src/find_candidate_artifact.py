from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from typing import Any


class CandidateArtifactError(RuntimeError):
    pass


def select_artifact(payload: dict[str, Any], expected_name: str) -> dict[str, Any]:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise CandidateArtifactError("GitHub artifact response is missing the artifacts list.")
    matches = [
        item
        for item in artifacts
        if isinstance(item, dict)
        and item.get("name") == expected_name
        and item.get("expired") is False
        and isinstance(item.get("workflow_run"), dict)
        and item["workflow_run"].get("id")
    ]
    if not matches:
        raise CandidateArtifactError(
            f"No non-expired immutable candidate artifact named {expected_name!r} was found."
        )
    matches.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    return matches[0]


def fetch_artifacts(repository: str, artifact_name: str, token: str) -> dict[str, Any]:
    encoded_name = urllib.parse.quote(artifact_name, safe="")
    url = (
        f"https://api.github.com/repos/{repository}/actions/artifacts"
        f"?name={encoded_name}&per_page=100"
    )
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "reid-release-pipeline",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise CandidateArtifactError("GitHub artifact response is not a JSON object.")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve the workflow run containing an immutable candidate artifact."
    )
    parser.add_argument("--repository", required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--token-env", default="GH_TOKEN")
    args = parser.parse_args()
    token = os.getenv(args.token_env, "")
    if not token:
        raise CandidateArtifactError(f"{args.token_env} is required to query GitHub artifacts.")
    artifact = select_artifact(
        fetch_artifacts(args.repository, args.artifact_name, token), args.artifact_name
    )
    print(artifact["workflow_run"]["id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
