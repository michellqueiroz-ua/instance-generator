import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (ROOT / ".github" / "workflows" / "docs.yml").read_text()


class DocsWorkflowTests(unittest.TestCase):
    def run_preflight(self, response, status=0):
        step = WORKFLOW.split("      - name: Check Pages configuration\n", 1)[1]
        block = step.split("        run: |\n", 1)[1].split("\n      - ", 1)[0]
        script = "\n".join(line[10:] for line in block.splitlines())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            result = subprocess.run(
                [
                    "bash", "-e", "-c",
                    'gh() { printf "%s\\n" "$MOCK_RESPONSE"; return "$MOCK_STATUS"; }\n'
                    + script,
                ],
                env={
                    **os.environ,
                    "GITHUB_OUTPUT": str(output),
                    "GITHUB_REPOSITORY": "owner/repo",
                    "MOCK_RESPONSE": response,
                    "MOCK_STATUS": str(status),
                },
                capture_output=True,
                text=True,
            )
            outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
        return result, outputs

    def test_actions_pages_enables_deployment(self):
        result, outputs = self.run_preflight("workflow")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(outputs["enabled"], "true")
        self.assertNotIn("::warning::", result.stdout)

    def test_missing_pages_skips_deployment_with_setup_warning(self):
        result, outputs = self.run_preflight("gh: Not Found (HTTP 404)", status=1)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(outputs["enabled"], "false")
        self.assertIn("::warning::Enable GitHub Pages", result.stdout)

    def test_branch_pages_skips_deployment_with_source_warning(self):
        result, outputs = self.run_preflight("legacy")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(outputs["enabled"], "false")
        self.assertIn("::warning::Select GitHub Actions", result.stdout)

    def test_unexpected_api_failure_is_not_hidden(self):
        for status in (403, 500):
            with self.subTest(status=status):
                response = f"gh: API failure (HTTP {status})"
                result, outputs = self.run_preflight(response, status=1)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(outputs["enabled"], "false")
                self.assertIn(response, result.stderr)

    def test_upload_and_deploy_require_enabled_pages(self):
        self.assertIn("pages: read", WORKFLOW)
        self.assertIn("pages_enabled: ${{ steps.pages.outputs.enabled }}", WORKFLOW)
        self.assertIn("if: steps.pages.outputs.enabled == 'true'", WORKFLOW)
        self.assertIn(
            "if: github.event_name == 'push' && github.ref == 'refs/heads/master'"
            " && needs.build.outputs.pages_enabled == 'true'",
            WORKFLOW,
        )


if __name__ == "__main__":
    unittest.main()
