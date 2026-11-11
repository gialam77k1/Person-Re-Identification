from __future__ import annotations

import unittest

from src.find_candidate_artifact import CandidateArtifactError, select_artifact
from src.resolve_release_publication import ReleasePublicationError, resolve_stage


class CandidateReleaseMaterializationTest(unittest.TestCase):
    def test_selects_latest_non_expired_exact_artifact(self) -> None:
        selected = select_artifact(
            {
                "artifacts": [
                    {
                        "name": "reports-cycle-1",
                        "expired": False,
                        "created_at": "2026-11-10T01:00:00Z",
                        "workflow_run": {"id": 10},
                    },
                    {
                        "name": "reports-cycle-1",
                        "expired": False,
                        "created_at": "2026-11-11T01:00:00Z",
                        "workflow_run": {"id": 11},
                    },
                    {
                        "name": "reports-cycle-1",
                        "expired": True,
                        "created_at": "2026-11-12T01:00:00Z",
                        "workflow_run": {"id": 12},
                    },
                ]
            },
            "reports-cycle-1",
        )
        self.assertEqual(selected["workflow_run"]["id"], 11)

    def test_rejects_missing_candidate_artifact(self) -> None:
        with self.assertRaises(CandidateArtifactError):
            select_artifact({"artifacts": []}, "reports-cycle-1")

    def test_publication_must_match_release_and_slug(self) -> None:
        publication = {
            "status": "PASS",
            "release_id": "market1501-duke-vit-bnneck:v2",
            "dvc": {"stage": "v2_package_triton"},
            "model_manifest": {
                "path": "artifacts/pipeline/market1501-duke-vit-bnneck-v2/release/model_manifest.json"
            },
        }
        self.assertEqual(
            resolve_stage(
                publication,
                "market1501-duke-vit-bnneck:v2",
                "market1501-duke-vit-bnneck-v2",
            ),
            "v2_package_triton",
        )
        with self.assertRaises(ReleasePublicationError):
            resolve_stage(publication, "market1501-duke-vit-bnneck:v3", "wrong-v3")


if __name__ == "__main__":
    unittest.main()
