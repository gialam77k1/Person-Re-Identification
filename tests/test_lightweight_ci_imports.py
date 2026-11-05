from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest


class LightweightCiImportsTest(unittest.TestCase):
    def test_numpy_only_modules_import_without_torch(self) -> None:
        program = textwrap.dedent(
            """
            import importlib.abc
            import sys

            class BlockTorch(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "torch" or fullname.startswith("torch."):
                        raise ModuleNotFoundError("torch intentionally blocked for lightweight CI")
                    return None

            sys.meta_path.insert(0, BlockTorch())
            import src.capture_external_embeddings
            import src.reid.cross_domain
            """
        )
        subprocess.run([sys.executable, "-c", program], check=True)


if __name__ == "__main__":
    unittest.main()
