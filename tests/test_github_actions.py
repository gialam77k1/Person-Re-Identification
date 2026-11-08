from __future__ import annotations

import unittest
import urllib.request

from src.control_api.github_actions import _CrossHostAuthRedirectHandler


class CrossHostAuthRedirectHandlerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.handler = _CrossHostAuthRedirectHandler()
        self.request = urllib.request.Request(
            "https://api.github.com/repos/example/project/actions/artifacts/1/zip",
            headers={
                "Authorization": "Bearer secret-token",
                "User-Agent": "person-reid-mlops-control-api",
            },
        )

    def redirect(self, target: str) -> urllib.request.Request:
        redirected = self.handler.redirect_request(
            self.request,
            None,
            302,
            "Found",
            {},
            target,
        )
        self.assertIsNotNone(redirected)
        return redirected

    def test_removes_authorization_when_artifact_redirects_to_storage(self) -> None:
        redirected = self.redirect(
            "https://results-receiver.actions.githubusercontent.com/artifact.zip?sig=abc"
        )

        self.assertIsNone(redirected.get_header("Authorization"))
        self.assertEqual(
            redirected.get_header("User-agent"),
            "person-reid-mlops-control-api",
        )

    def test_keeps_authorization_for_same_origin_redirect(self) -> None:
        redirected = self.redirect(
            "https://api.github.com/repos/example/project/actions/artifacts/1/archive"
        )

        self.assertEqual(
            redirected.get_header("Authorization"),
            "Bearer secret-token",
        )


if __name__ == "__main__":
    unittest.main()
