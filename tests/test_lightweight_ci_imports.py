from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest


class LightweightCiImportsTest(unittest.TestCase):
    def test_numpy_only_modules_import_without_training_dependencies(self) -> None:
        program = textwrap.dedent(
            """
            import importlib.abc
            import sys

            class BlockTrainingDependencies(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    blocked = ("torch", "tqdm")
                    if fullname in blocked or fullname.startswith(tuple(f"{name}." for name in blocked)):
                        raise ModuleNotFoundError(
                            f"{fullname} intentionally blocked for lightweight CI"
                        )
                    return None

            sys.meta_path.insert(0, BlockTrainingDependencies())
            import src.capture_external_embeddings
            import src.reid.cross_domain
            """
        )
        subprocess.run([sys.executable, "-c", program], check=True)


if __name__ == "__main__":
    unittest.main()
