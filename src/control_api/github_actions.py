from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from src.control_api.config import ControlSettings


class WorkflowDispatchError(RuntimeError):
    pass


class GitHubActionsClient:
    def __init__(self, settings: ControlSettings) -> None:
        self.settings = settings

    def dispatch(self, workflow: str, inputs: dict[str, str]) -> dict[str, Any]:
        if not self.settings.github_token:
            raise WorkflowDispatchError(
                "CONTROL_GITHUB_TOKEN is required to dispatch GitHub Actions workflows."
            )
        repository = urllib.parse.quote(self.settings.github_repository, safe="/")
        workflow_name = urllib.parse.quote(workflow, safe="")
        url = (
            f"{self.settings.github_api_url}/repos/{repository}/actions/workflows/"
            f"{workflow_name}/dispatches"
        )
        request = urllib.request.Request(
            url,
            method="POST",
            data=json.dumps({"ref": self.settings.workflow_ref, "inputs": inputs}).encode("utf-8"),
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.settings.github_token}",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "person-reid-mlops-control-api",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                if response.status != 204:
                    raise WorkflowDispatchError(
                        f"GitHub workflow dispatch returned HTTP {response.status}."
                    )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise WorkflowDispatchError(
                f"GitHub workflow dispatch returned HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise WorkflowDispatchError(f"Cannot reach GitHub Actions: {exc}") from exc
        return {"workflow": workflow, "ref": self.settings.workflow_ref, "inputs": inputs}
