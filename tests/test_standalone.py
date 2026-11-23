"""Read-only smoke tests for the extracted frozen killfeed detector."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "rev12"
sys.path.insert(0, str(CORE))
import entrypoint


class StandaloneSmokeTests(unittest.TestCase):
    def test_unmodified_rev12_integrity_manifest(self):
        self.assertEqual(
            entrypoint.verify(),
            "2e3c7a1931c21833712156a44bbe4d9cff21b746b6ab0ee63c2632a1740087de"
        )
        records = json.loads((CORE / "REV12_SOURCE_SHA256.json").read_text())
        self.assertEqual(len(records), 10)
        self.assertFalse(any("v14_bridge" in name for name in records))

    def test_public_cli_help(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "detect.py"), "--help"],
            capture_output=True, text=True
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--owner", proc.stdout)

    def test_public_cli_requires_owner(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "detect.py"),
             "--input", "anything.mp4", "--output", "out"],
            capture_output=True, text=True
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--owner", proc.stderr)

    def test_public_cli_rejects_missing_source_without_creating_output(self):
        with tempfile.TemporaryDirectory() as t:
            target = Path(t) / "result"
            proc = subprocess.run(
                [sys.executable, str(ROOT / "detect.py"),
                 "--input", str(Path(t) / "no.mp4"), "--output", str(target),
                 "--owner", "Player"],
                capture_output=True, text=True
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
