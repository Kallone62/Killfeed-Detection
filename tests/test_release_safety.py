"""Offline boundary tests for the independently published detector source."""
import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [ROOT / "detect.py", *(ROOT / "rev12").rglob("*.py")]
FORBIDDEN_MODULES = {
    "requests", "httpx", "urllib", "socket", "boto3",
    "firebase_admin", "google.cloud", "paramiko", "smtplib"
}
FORBIDDEN_CONTENT = re.compile(
    r"https?://|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}|"
    r"gh[pousr]_[A-Za-z0-9_]{35,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"(?i:Kallone62|descripcs2-automation|descripcs2\.com|"
    r"GCP_PROJECT_ID|AWS_SECRET_ACCESS_KEY|CLIENT_SECRET|"
    r"ACCESS_TOKEN|REFRESH_TOKEN|API_KEY)"
)


class PublicSourceBoundaryTests(unittest.TestCase):
    def test_no_network_or_cloud_client_imports(self):
        for src in SOURCES:
            tree = ast.parse(src.read_text(encoding="utf-8"))
            modules = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules.extend(alias.name for alias in node.names)
                if isinstance(node, ast.ImportFrom) and node.module:
                    modules.append(node.module)
            for name in modules:
                with self.subTest(src=src.name, module=name):
                    self.assertFalse(
                        any(name == mod or name.startswith(mod + ".") for mod in FORBIDDEN_MODULES),
                        f"Forbidden network/cloud import in {src}: {name}"
                    )

    def test_no_credential_markers_private_urls_or_project_identifiers(self):
        for src in SOURCES:
            with self.subTest(src=src.name):
                self.assertIsNone(
                    FORBIDDEN_CONTENT.search(src.read_text(encoding="utf-8")),
                    f"Review sensitive data in {src}"
                )


if __name__ == "__main__":
    unittest.main()
