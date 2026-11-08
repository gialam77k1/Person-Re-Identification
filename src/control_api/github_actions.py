from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from io import BytesIO
from typing import Any

from src.control_api.config import ControlSettings


class WorkflowDispatchError(RuntimeError):
    pass


class _CrossHostAuthRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Do not forward the GitHub bearer token to artifact storage hosts."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None
        source = urllib.parse.urlsplit(req.full_url)
        destination = urllib.parse.urlsplit(newurl)
        if (source.scheme, source.netloc) != (destination.scheme, destination.netloc):
            redirected.remove_header("Authorization")
        return redirected


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

    def download_artifact(self, run_id: str, artifact_name: str) -> dict[str, bytes]:
        if not self.settings.github_token:
            raise WorkflowDispatchError(
                "CONTROL_GITHUB_TOKEN is required to download workflow evidence."
            )
        repository = urllib.parse.quote(self.settings.github_repository, safe="/")
        listing = self._request_json(
            f"{self.settings.github_api_url}/repos/{repository}/actions/runs/"
            f"{urllib.parse.quote(str(run_id), safe='')}/artifacts"
        )
        matches = [
            item
            for item in listing.get("artifacts", [])
            if item.get("name") == artifact_name and not item.get("expired", False)
        ]
        if len(matches) != 1:
            raise WorkflowDispatchError(
                f"Expected one active artifact named {artifact_name!r}, found {len(matches)}."
            )
        archive = self._request_bytes(str(matches[0]["archive_download_url"]))
        try:
            with zipfile.ZipFile(BytesIO(archive)) as bundle:
                files = {
                    name: bundle.read(name)
                    for name in bundle.namelist()
                    if not name.endswith("/")
                }
        except zipfile.BadZipFile as exc:
            raise WorkflowDispatchError("GitHub evidence artifact is not a valid ZIP archive.") from exc
        if not files:
            raise WorkflowDispatchError("GitHub evidence artifact is empty.")
        return files

    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.settings.github_token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "person-reid-mlops-control-api",
        }

    def _request_json(self, url: str) -> dict[str, Any]:
        return json.loads(self._request_bytes(url).decode("utf-8"))

    def _request_bytes(self, url: str) -> bytes:
        request = urllib.request.Request(url, headers=self._headers())
        opener = urllib.request.build_opener(_CrossHostAuthRedirectHandler())
        try:
            with opener.open(request, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise WorkflowDispatchError(
                f"GitHub artifact request returned HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise WorkflowDispatchError(f"Cannot download GitHub evidence: {exc}") from exc
