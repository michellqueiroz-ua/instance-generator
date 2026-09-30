import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PackagingTests(unittest.TestCase):
    def test_requirements_installs_app_extra_from_package(self):
        requirements = (ROOT / "requirements.txt").read_text().splitlines()
        active_lines = [
            line.strip()
            for line in requirements
            if line.strip() and not line.lstrip().startswith("#")
        ]

        self.assertEqual(active_lines, ["-e .[app]"])

    def test_conda_environment_installs_app_extra_from_package(self):
        environment = (
            ROOT / "environment" / "REQreate_environment.yml"
        ).read_text().splitlines()
        dependencies_index = environment.index("dependencies:")
        dependencies = []
        pip_dependencies = []
        in_pip_dependencies = False

        for line in environment[dependencies_index + 1 :]:
            if line.startswith("  - "):
                in_pip_dependencies = line.strip() == "- pip:"
                dependencies.append(line.strip())
            elif in_pip_dependencies and line.startswith("      - "):
                pip_dependencies.append(line.strip())

        self.assertEqual(
            dependencies,
            ["- python=3.11", "- pip", "- pip:"],
        )
        self.assertEqual(pip_dependencies, ['- "-e .[app]"'])


if __name__ == "__main__":
    unittest.main()
