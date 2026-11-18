from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from src.qdrant_release_index import (
    QdrantIndexError,
    QdrantReleaseIndex,
    environment_alias,
    load_reference_bundle,
    release_collection_name,
)


class FakeQdrant:
    def __init__(self) -> None:
        self.collections: dict[str, dict] = {}
        self.aliases: dict[str, str] = {}

    def request(self, method: str, url: str, payload: dict | None = None) -> dict:
        parsed = urlparse(url)
        parts = [part for part in parsed.path.split("/") if part]
        if method == "GET" and parts == ["aliases"]:
            return {
                "result": {
                    "aliases": [
                        {"alias_name": alias, "collection_name": collection}
                        for alias, collection in self.aliases.items()
                    ]
                }
            }
        if method == "POST" and parts == ["collections", "aliases"]:
            for action in (payload or {}).get("actions", []):
                if "delete_alias" in action:
                    self.aliases.pop(action["delete_alias"]["alias_name"], None)
                if "create_alias" in action:
                    value = action["create_alias"]
                    self.aliases[value["alias_name"]] = value["collection_name"]
            return {"result": True}
        if len(parts) >= 2 and parts[0] == "collections":
            name = parts[1]
            if method == "PUT" and len(parts) == 2:
                self.collections[name] = {
                    "dimension": payload["vectors"]["size"],
                    "points": {},
                }
                return {"result": True}
            if method == "DELETE" and len(parts) == 2:
                self.collections.pop(name, None)
                return {"result": True}
            if method == "GET" and len(parts) == 2:
                if name not in self.collections:
                    raise QdrantIndexError("Qdrant returned HTTP 404: not found")
                value = self.collections[name]
                return {
                    "result": {
                        "status": "green",
                        "points_count": len(value["points"]),
                        "config": {
                            "params": {"vectors": {"size": value["dimension"]}}
                        },
                    }
                }
            if method == "PUT" and parts[2:] == ["points"]:
                for point in payload["points"]:
                    self.collections[name]["points"][point["id"]] = point
                return {"result": True}
            if method == "POST" and parts[2:] == ["points", "scroll"]:
                points = list(self.collections[name]["points"].values())
                return {
                    "result": {
                        "points": [
                            {"id": point["id"], "payload": point["payload"]}
                            for point in points[: payload["limit"]]
                        ]
                    }
                }
        raise AssertionError(f"Unexpected Qdrant request: {method} {url} {payload}")


class QdrantReleaseIndexTest(unittest.TestCase):
    def make_bundle(self, root: Path, release_id: str = "person-reid:v2"):
        embeddings = np.eye(3, 4, dtype=np.float32)
        arrays = {
            "reference_embeddings": embeddings,
            "reference_pids": np.asarray([10, 11, 12], dtype=np.int64),
            "reference_camids": np.asarray([1, 2, 3], dtype=np.int64),
            "reference_paths": np.asarray(["a.jpg", "b.jpg", "c.jpg"]),
        }
        files = []
        for role, value in arrays.items():
            path = root / f"{role}.npy"
            np.save(path, value, allow_pickle=False)
            files.append(
                {
                    "role": role,
                    "repository_path": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        manifest = {
            "release": {"release_id": release_id},
            "source": {"git_commit": "a" * 40},
            "validation": {"status": "PASS"},
            "contract": {"embedding_dim": 4},
            "files": files,
        }
        path = root / "model_manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path, load_reference_bundle(path, release_id, root=root)

    def test_collection_name_and_alias_are_version_neutral(self) -> None:
        checksum = "a" * 64
        self.assertIn("person_reid_v3", release_collection_name("person-reid:v3", checksum))
        self.assertEqual(environment_alias("production"), "reid_reference_active")
        self.assertEqual(environment_alias("staging"), "reid_reference_staging")

    def test_builds_immutable_collection_and_switches_alias_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, bundle = self.make_bundle(Path(directory))
            fake = FakeQdrant()
            manager = QdrantReleaseIndex(
                "http://qdrant:6333", request=fake.request, batch_size=2
            )
            collection = manager.ensure_collection(bundle)
            self.assertEqual(manager.ensure_collection(bundle), collection)
            self.assertEqual(len(fake.collections[collection]["points"]), 3)
            first = fake.collections[collection]["points"][0]
            self.assertEqual(first["payload"]["release_id"], "person-reid:v2")

            self.assertIsNone(manager.switch_alias("reid_reference_active", collection))
            self.assertEqual(fake.aliases["reid_reference_active"], collection)
            manager.restore_alias("reid_reference_active", None)
            self.assertNotIn("reid_reference_active", fake.aliases)

    def test_rejects_tampered_reference_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self.make_bundle(root)
            (root / "reference_pids.npy").write_bytes(b"tampered")
            with self.assertRaises(QdrantIndexError):
                load_reference_bundle(manifest, "person-reid:v2", root=root)


if __name__ == "__main__":
    unittest.main()
