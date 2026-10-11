"""Code-release smoke tests that do not require retained experiment artifacts."""
import importlib
from pathlib import Path
import unittest


class PublicReleaseTests(unittest.TestCase):
    def test_all_package_modules_import(self):
        package = Path(__file__).resolve().parents[1] / "incontext_collapse"
        for source in sorted(package.glob("*.py")):
            if source.name == "__init__.py":
                continue
            with self.subTest(module=source.stem):
                importlib.import_module(f"incontext_collapse.{source.stem}")

    def test_no_private_release_directories(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("paper", "data", "figures", "reproducibility"):
            self.assertFalse((root / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
