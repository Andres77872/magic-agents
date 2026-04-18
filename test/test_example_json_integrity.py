"""
Approval tests for JSON example file integrity.

Tests that scan all JSON files under examples/ to ensure:
- No hardcoded API key patterns exist (sk-proj-, jina_, etc.)
- Files that should use env placeholders have {{env. patterns
- All JSON files are valid and parseable
"""
import json
import os
from pathlib import Path

import pytest


# Patterns that indicate hardcoded API keys
HARDCODED_KEY_PATTERNS = [
    "sk-proj-",    # OpenAI project keys
    "sk-",         # OpenAI keys (general)
    "jina_",       # Jina API keys
]

# The expected env placeholder pattern used by build()
ENV_PLACEHOLDER_PATTERN = "{{env."


def _find_example_json_files() -> list[Path]:
    """Find all JSON files under examples/."""
    project_root = Path(__file__).parent.parent
    examples_dir = project_root / "examples"
    if not examples_dir.exists():
        return []
    return sorted(examples_dir.rglob("*.json"))


class TestNoHardcodedApiKeys:
    """Ensure no hardcoded API keys exist in example JSON files."""

    def test_no_hardcoded_api_keys_in_examples(self):
        """Scan all JSON files under examples/ for hardcoded key patterns."""
        json_files = _find_example_json_files()
        if not json_files:
            pytest.skip("No JSON files found in examples/")

        violations = []
        for json_file in json_files:
            content = json_file.read_text()
            for pattern in HARDCODED_KEY_PATTERNS:
                if pattern in content:
                    # Find which line(s) contain the pattern
                    for line_num, line in enumerate(content.splitlines(), 1):
                        if pattern in line:
                            violations.append(
                                f"{json_file.relative_to(Path(__file__).parent.parent)}:"
                                f"{line_num}: contains '{pattern}'"
                            )

        if violations:
            pytest.fail(
                f"Found hardcoded API key patterns in {len(violations)} location(s):\n"
                + "\n".join(violations)
            )

    def test_all_example_jsons_are_valid_json(self):
        """Every JSON file under examples/ must be valid JSON."""
        json_files = _find_example_json_files()
        if not json_files:
            pytest.skip("No JSON files found in examples/")

        invalid_files = []
        for json_file in json_files:
            try:
                with open(json_file, "r") as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                invalid_files.append(f"{json_file.name}: {e}")

        if invalid_files:
            pytest.fail(
                f"Found {len(invalid_files)} invalid JSON file(s):\n"
                + "\n".join(invalid_files)
            )


class TestEnvPlaceholders:
    """Ensure env placeholder patterns are used where expected."""

    def test_json_files_with_client_nodes_use_env_placeholders(self):
        """JSON files that have client nodes should use {{env.OPENAI_API_KEY}}."""
        json_files = _find_example_json_files()
        if not json_files:
            pytest.skip("No JSON files found in examples/")

        files_with_client_no_placeholder = []
        for json_file in json_files:
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                continue  # Already caught by the valid JSON test

            # Check both flat and wrapped variants
            nodes = data.get("nodes", [])
            if not nodes:
                nodes = data.get("content", {}).get("nodes", [])

            has_client = any(n.get("type") == "client" for n in nodes)
            has_placeholder = ENV_PLACEHOLDER_PATTERN in json_file.read_text()

            if has_client and not has_placeholder:
                files_with_client_no_placeholder.append(
                    json_file.relative_to(Path(__file__).parent.parent)
                )

        if files_with_client_no_placeholder:
            pytest.fail(
                f"JSON files with client nodes but no env placeholders:\n"
                + "\n".join(str(f) for f in files_with_client_no_placeholder)
            )
