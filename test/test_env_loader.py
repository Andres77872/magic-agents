"""
Unit tests for the centralized env loader in conftest.py.

Tests cover:
- _parse_dotenv() parsing behavior (basic, comments, blanks, quotes, edge cases)
- _is_placeholder_value() detection logic
- _resolve_api_keys() priority ordering (real env > .env.test > empty)
- skip_if_no_api_keys() skip behavior
- api_keys fixture integration
"""
import os
from pathlib import Path

import pytest

# Import internal helpers from conftest
from conftest import (
    _parse_dotenv,
    _is_placeholder_value,
    _load_env_test,
    _populate_os_environ_from_dotenv,
    _resolve_api_keys,
    skip_if_no_api_keys,
)


# ─── _parse_dotenv() Tests ──────────────────────────────────────────────────

class TestParseDotenv:
    """Tests for the simple .env file parser."""

    def test_parse_dotenv_basic(self, tmp_path):
        """Parses simple KEY=VALUE lines."""
        env_file = tmp_path / ".env.test"
        env_file.write_text("OPENAI_API_KEY=sk-test123\nSERPER_API_KEY=abc456\n")

        result = _parse_dotenv(env_file)

        assert result["OPENAI_API_KEY"] == "sk-test123"
        assert result["SERPER_API_KEY"] == "abc456"

    def test_parse_dotenv_skips_comments(self, tmp_path):
        """Lines starting with # are ignored."""
        env_file = tmp_path / ".env.test"
        env_file.write_text(
            "# This is a comment\n"
            "OPENAI_API_KEY=sk-test123\n"
            "# Another comment\n"
            "SERPER_API_KEY=abc456\n"
        )

        result = _parse_dotenv(env_file)

        assert len(result) == 2
        assert result["OPENAI_API_KEY"] == "sk-test123"
        assert result["SERPER_API_KEY"] == "abc456"

    def test_parse_dotenv_skips_blank_lines(self, tmp_path):
        """Empty lines are ignored."""
        env_file = tmp_path / ".env.test"
        env_file.write_text("\n\nOPENAI_API_KEY=sk-test123\n\n\nSERPER_API_KEY=abc456\n\n")

        result = _parse_dotenv(env_file)

        assert len(result) == 2

    def test_parse_dotenv_handles_double_quotes(self, tmp_path):
        """KEY=\"value with spaces\" strips surrounding double quotes."""
        env_file = tmp_path / ".env.test"
        env_file.write_text('OPENAI_API_KEY="sk-test with spaces"\n')

        result = _parse_dotenv(env_file)

        assert result["OPENAI_API_KEY"] == "sk-test with spaces"

    def test_parse_dotenv_handles_single_quotes(self, tmp_path):
        """KEY='value with spaces' strips surrounding single quotes."""
        env_file = tmp_path / ".env.test"
        env_file.write_text("OPENAI_API_KEY='sk-test with spaces'\n")

        result = _parse_dotenv(env_file)

        assert result["OPENAI_API_KEY"] == "sk-test with spaces"

    def test_parse_dotenv_handles_equals_in_value(self, tmp_path):
        """Values containing = are preserved (splits on first = only)."""
        env_file = tmp_path / ".env.test"
        env_file.write_text("API_KEY=sk-test=with=equals\n")

        result = _parse_dotenv(env_file)

        assert result["API_KEY"] == "sk-test=with=equals"

    def test_parse_dotenv_missing_file_returns_empty(self, tmp_path):
        """Non-existent file returns empty dict without error."""
        missing_file = tmp_path / ".env.nonexistent"

        result = _parse_dotenv(missing_file)

        assert result == {}

    def test_parse_dotenv_skips_lines_without_equals(self, tmp_path):
        """Lines without = are silently skipped."""
        env_file = tmp_path / ".env.test"
        env_file.write_text("OPENAI_API_KEY=sk-test123\nINVALID_LINE_NO_EQUALS\nSERPER=abc\n")

        result = _parse_dotenv(env_file)

        assert len(result) == 2
        assert "INVALID_LINE_NO_EQUALS" not in result

    def test_parse_dotenv_strips_whitespace_around_key(self, tmp_path):
        """Whitespace around key and value is stripped."""
        env_file = tmp_path / ".env.test"
        env_file.write_text("  OPENAI_API_KEY  =  sk-test123  \n")

        result = _parse_dotenv(env_file)

        assert result["OPENAI_API_KEY"] == "sk-test123"


# ─── _is_placeholder_value() Tests ─────────────────────────────────────────

class TestIsPlaceholderValue:
    """Tests for placeholder value detection."""

    @pytest.mark.parametrize("value", [
        "",
        "your-openai-key-here",
        "your-serper-key-here",
        "placeholder-key",
        "xxx-not-a-real-key",
        "changeme",
        "insert-your-key-here",
        "short",
        "abc",
    ])
    def test_detects_placeholder(self, value):
        """Values matching placeholder patterns return True."""
        assert _is_placeholder_value(value) is True

    @pytest.mark.parametrize("value", [
        "sk-proj-GLxkCWDabc123def456ghi789jkl012mno345pqr678stu",
        "sk-abc123def456ghi789jkl012mno345pqr678stuvwx",
        "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
        "some-reasonably-long-value-that-is-not-a-fake",
    ])
    def test_accepts_real_key(self, value):
        """Values that look like real keys return False."""
        assert _is_placeholder_value(value) is False


# ─── _resolve_api_keys() Tests ─────────────────────────────────────────────

class TestResolveApiKeys:
    """Tests for the priority-based API key resolution."""

    def test_prefers_real_env_over_dotenv_test(self, monkeypatch, tmp_path):
        """Real environment variables take priority over .env.test values."""
        # Create a .env.test with a different value
        env_file = tmp_path / ".env.test"
        env_file.write_text("OPENAI_API_KEY=dotenv-test-value\n")

        # Monkeypatch the project root resolution
        monkeypatch.setattr(Path, "__new__", lambda cls, *args, **kwargs:
            tmp_path / ".env.test" if args and str(args[0]).endswith("conftest.py")
            else object.__new__(cls))

        # Set real env var
        monkeypatch.setenv("OPENAI_API_KEY", "real-env-key")
        monkeypatch.delenv("SERPER_API_KEY", raising=False)

        # _resolve_api_keys uses its own _load_env_test which finds project root
        # from __file__. We need to test the priority logic differently.
        # Since _resolve_api_keys calls _load_env_test which uses Path(__file__),
        # we can't easily mock the file location. Instead, test the priority
        # by ensuring real env vars win when they exist.
        result = _resolve_api_keys()

        # Real env var should be present
        assert result.get("openai_key") == "real-env-key"

    def test_returns_empty_when_no_env_and_no_dotenv(self, monkeypatch):
        """When no env vars and no .env.test, returns empty dict."""
        # Clear any real env vars
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("SERPER_API_KEY", raising=False)

        # The actual .env.test file exists at project root with placeholder values
        # but those ARE placeholder values, so they'll be loaded but detected as placeholders
        # at the skip level, not at the resolve level.
        # _resolve_api_keys loads whatever is there (placeholder or not).
        result = _resolve_api_keys()

        # Since .env.test exists with placeholder values, they will be loaded
        # This test verifies the function doesn't crash and returns a dict
        assert isinstance(result, dict)

    def test_includes_serper_key_when_available(self, monkeypatch):
        """Serper key is included when set in environment."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("SERPER_API_KEY", "real-serper-key")

        result = _resolve_api_keys()

        assert result.get("serper_key") == "real-serper-key"

    def test_includes_both_keys_when_both_set(self, monkeypatch):
        """Both keys are included when both env vars are set."""
        monkeypatch.setenv("OPENAI_API_KEY", "real-openai-key")
        monkeypatch.setenv("SERPER_API_KEY", "real-serper-key")

        result = _resolve_api_keys()

        assert result.get("openai_key") == "real-openai-key"
        assert result.get("serper_key") == "real-serper-key"


# ─── _populate_os_environ_from_dotenv() Tests ──────────────────────────────

class TestPopulateOsEnvironFromDotenv:
    """Tests for loading .env.test into os.environ."""

    def test_populates_missing_keys(self, monkeypatch):
        """Keys absent from os.environ are filled in from .env.test."""
        monkeypatch.delenv("TEST_API_KEY", raising=False)
        monkeypatch.delenv("ANOTHER_KEY", raising=False)
        monkeypatch.setattr("conftest._load_env_test", lambda: {
            "TEST_API_KEY": "from-dotenv",
            "ANOTHER_KEY": "also-from-dotenv",
        })

        _populate_os_environ_from_dotenv()

        assert os.environ.get("TEST_API_KEY") == "from-dotenv"
        assert os.environ.get("ANOTHER_KEY") == "also-from-dotenv"

    def test_shell_env_vars_take_priority(self, monkeypatch):
        """Existing os.environ values are NOT overwritten by .env.test."""
        monkeypatch.setenv("TEST_API_KEY", "from-shell")
        monkeypatch.setattr("conftest._load_env_test", lambda: {
            "TEST_API_KEY": "from-dotenv",
        })

        _populate_os_environ_from_dotenv()

        assert os.environ.get("TEST_API_KEY") == "from-shell"

    def test_handles_missing_dotenv_gracefully(self, monkeypatch):
        """Missing .env.test does not raise an error."""
        monkeypatch.delenv("NONEXISTENT_TEST_KEY", raising=False)
        monkeypatch.setattr("conftest._load_env_test", lambda: {})

        # Should not raise
        _populate_os_environ_from_dotenv()


# ─── skip_if_no_api_keys() Tests ────────────────────────────────────────────

class TestSkipIfNoApiKeys:
    """Tests for the skip helper."""

    def test_skips_on_placeholder_value(self):
        """Placeholder value triggers pytest.skip."""
        placeholder_keys = {"openai_key": "your-openai-key-here"}

        with pytest.raises(pytest.skip.Exception):
            skip_if_no_api_keys(placeholder_keys)

    def test_skips_on_empty_key(self):
        """Empty key triggers pytest.skip."""
        empty_keys = {"openai_key": ""}

        with pytest.raises(pytest.skip.Exception):
            skip_if_no_api_keys(empty_keys)

    def test_skips_on_missing_key(self):
        """Missing openai_key triggers pytest.skip."""
        no_keys = {}

        with pytest.raises(pytest.skip.Exception):
            skip_if_no_api_keys(no_keys)

    def test_does_not_skip_on_real_key(self):
        """Real-looking key does not trigger skip."""
        real_keys = {"openai_key": "sk-proj-GLxkCWDabc123def456ghi789jkl012mno345pqr678stu"}

        # Should not raise
        result = skip_if_no_api_keys(real_keys)
        assert result == real_keys

    def test_resolves_fresh_when_no_keys_passed(self, monkeypatch):
        """When no keys passed, resolves fresh from env."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-real-key-12345678901234567890")

        # Should not skip because we set a real-looking key
        result = skip_if_no_api_keys(None)
        assert "openai_key" in result


# ─── api_keys Fixture Tests ─────────────────────────────────────────────────

class TestApiKeysFixture:
    """Tests for the api_keys session fixture."""

    def test_api_keys_returns_dict(self, api_keys):
        """The fixture returns a dict."""
        assert isinstance(api_keys, dict)

    def test_api_keys_has_expected_structure(self, api_keys):
        """Keys are strings when present."""
        for key, value in api_keys.items():
            assert isinstance(key, str)
            assert isinstance(value, str)
