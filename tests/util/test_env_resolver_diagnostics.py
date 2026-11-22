"""
Unit tests for env_resolver diagnostic logging (missing-var warnings).

Tests cover the ``logger.warning`` behavior added in Phase 1.5 of
fix-deep-research-tool-errors:

- Missing ``{{env.VAR}}`` → ``logger.warning`` called with the variable name only
- Present env var → ``logger.warning`` NOT called
- Multiple missing vars in one string → one warning per missing var
- Non-``{{env.*}}`` placeholders (e.g. ``{{name}}``) → no warning triggered
- No actual env var values appear in log output arguments

All tests use ``unittest.mock.patch`` to capture ``logger.warning`` calls and
``monkeypatch`` to control ``os.environ``, so no real env state leaks between
tests.
"""
from unittest.mock import patch

import pytest

from magic_agents.util.env_resolver import resolve_env_string


# ─── Missing Env Var Warning ─────────────────────────────────────────────────


class TestMissingEnvVarWarning:
    """logger.warning is called when {{env.VAR}} resolves to empty string."""

    def test_missing_var_logs_warning_with_var_name(self, monkeypatch):
        """Missing env var triggers logger.warning with the variable name only."""
        monkeypatch.delenv("TEST_MISSING_VAR", raising=False)

        with patch("magic_agents.util.env_resolver.logger.warning") as mock_warn:
            result = resolve_env_string("{{env.TEST_MISSING_VAR}}")

        assert result == ""
        mock_warn.assert_called_once()
        args, _ = mock_warn.call_args
        # logger.warning uses %s formatting — var_name is in args[1] and args[2]
        assert args[1] == "TEST_MISSING_VAR"
        assert args[2] == "TEST_MISSING_VAR"

    def test_missing_var_warning_does_not_contain_value(self, monkeypatch):
        """The log message MUST NOT contain any env var value or secret."""
        monkeypatch.delenv("TEST_MISSING_VAR", raising=False)

        with patch("magic_agents.util.env_resolver.logger.warning") as mock_warn:
            resolve_env_string("{{env.TEST_MISSING_VAR}}")

        mock_warn.assert_called_once()
        args, _ = mock_warn.call_args
        log_message = str(args)
        # The var name itself is fine, but we should not see any value
        # associated with it. Since the var is empty, there's no value,
        # but the pattern should prevent future regressions.
        assert "sk-" not in log_message
        assert "api_key" not in log_message.lower()

    def test_missing_var_returns_empty_string(self, monkeypatch):
        """Return value is the empty string (behavior unchanged from before)."""
        monkeypatch.delenv("TEST_MISSING_VAR_RETURN", raising=False)

        with patch("magic_agents.util.env_resolver.logger.warning"):
            result = resolve_env_string("prefix{{env.TEST_MISSING_VAR_RETURN}}suffix")

        assert result == "prefixsuffix"


# ─── Present Env Var (No Warning) ────────────────────────────────────────────


class TestPresentEnvVarNoWarning:
    """logger.warning is NOT called when {{env.VAR}} resolves to an actual value."""

    def test_present_var_does_not_log_warning(self, monkeypatch):
        """Present env var does NOT trigger logger.warning."""
        monkeypatch.setenv("TEST_PRESENT_VAR", "test-value-123")

        with patch("magic_agents.util.env_resolver.logger.warning") as mock_warn:
            result = resolve_env_string("{{env.TEST_PRESENT_VAR}}")

        assert result == "test-value-123"
        mock_warn.assert_not_called()

    def test_present_var_returns_actual_value(self, monkeypatch):
        """Return value is the resolved env var value (behavior unchanged)."""
        monkeypatch.setenv("TEST_PRESENT_VAR_VAL", "sk-real-value")

        with patch("magic_agents.util.env_resolver.logger.warning"):
            result = resolve_env_string("{{env.TEST_PRESENT_VAR_VAL}}")

        assert result == "sk-real-value"

    def test_present_var_in_context_no_warning(self, monkeypatch):
        """A string with both present and non-env placeholders: no warning."""
        monkeypatch.setenv("API_KEY", "abc123")

        with patch("magic_agents.util.env_resolver.logger.warning") as mock_warn:
            result = resolve_env_string("Bearer {{env.API_KEY}} :: {{handle_fetch_input}}")

        assert result == "Bearer abc123 :: {{handle_fetch_input}}"
        mock_warn.assert_not_called()


# ─── Multiple Missing Vars ───────────────────────────────────────────────────


class TestMultipleMissingVars:
    """Multiple missing {{env.VAR}} placeholders each trigger a warning."""

    def test_multiple_missing_vars_warn_per_placeholder(self, monkeypatch):
        """Each missing var triggers its own warning."""
        monkeypatch.delenv("KEY_A", raising=False)
        monkeypatch.delenv("KEY_B", raising=False)

        with patch("magic_agents.util.env_resolver.logger.warning") as mock_warn:
            result = resolve_env_string("{{env.KEY_A}}:{{env.KEY_B}}")

        assert result == ":"
        assert mock_warn.call_count >= 2
        # logger.warning uses %s formatting — var_name is in args[1] and args[2]
        call_args_list = [args for args, _ in mock_warn.call_args_list]
        var_names = [args[1] for args in call_args_list]
        assert "KEY_A" in var_names
        assert "KEY_B" in var_names

    def test_single_missing_and_present_var(self, monkeypatch):
        """Only the missing var triggers a warning, not the present one."""
        monkeypatch.setenv("KEY_PRESENT", "real-value")
        monkeypatch.delenv("KEY_MISSING", raising=False)

        with patch("magic_agents.util.env_resolver.logger.warning") as mock_warn:
            result = resolve_env_string("{{env.KEY_MISSING}}/{{env.KEY_PRESENT}}")

        assert result == "/real-value"
        mock_warn.assert_called_once()
        args, _ = mock_warn.call_args
        # logger.warning uses %s formatting — var_name is in args[1] and args[2]
        assert args[1] == "KEY_MISSING"
        assert args[2] == "KEY_MISSING"

    def test_three_missing_vars(self, monkeypatch):
        """Three missing vars trigger three warnings."""
        for var in ["A1", "B2", "C3"]:
            monkeypatch.delenv(var, raising=False)

        with patch("magic_agents.util.env_resolver.logger.warning") as mock_warn:
            result = resolve_env_string("{{env.A1}}-{{env.B2}}-{{env.C3}}")

        assert result == "--"
        assert mock_warn.call_count == 3


# ─── Non-env Placeholders (No Warning) ──────────────────────────────────────


class TestNonEnvPlaceholders:
    """Non-{{env.*}} placeholders do NOT trigger warnings."""

    def test_non_env_placeholder_no_warning(self, monkeypatch):
        """A Jinja2-style {{name}} placeholder triggers no warning."""
        with patch("magic_agents.util.env_resolver.logger.warning") as mock_warn:
            result = resolve_env_string("Hello {{name}}")

        assert result == "Hello {{name}}"
        mock_warn.assert_not_called()

    @pytest.mark.parametrize("input_str,expected", [
        ("{{name}}", "{{name}}"),
        ("{{user.id}}", "{{user.id}}"),
        ("{{ handle_fetch_input }}", "{{ handle_fetch_input }}"),
        ("no placeholders at all", "no placeholders at all"),
        ("{{ something_else }}", "{{ something_else }}"),
    ])
    def test_various_non_env_patterns_no_warning(self, input_str, expected):
        """Various non-env placeholder patterns produce no warning."""
        with patch("magic_agents.util.env_resolver.logger.warning") as mock_warn:
            result = resolve_env_string(input_str)

        assert result == expected
        mock_warn.assert_not_called()

    def test_mixed_env_and_non_env_no_extra_warnings(self, monkeypatch):
        """A string with both missing env vars and non-env placeholders
        warns only for the missing env vars, not the non-env ones."""
        monkeypatch.delenv("API_KEY_MIXED", raising=False)

        with patch("magic_agents.util.env_resolver.logger.warning") as mock_warn:
            result = resolve_env_string(
                "{{env.API_KEY_MIXED}} and {{ display_name }}"
            )

        assert result == " and {{ display_name }}"
        mock_warn.assert_called_once()
        args, _ = mock_warn.call_args
        # logger.warning uses %s formatting — var_name is in args[1] and args[2]
        assert args[1] == "API_KEY_MIXED"


# ─── Edge Cases ──────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases for env_resolver diagnostic warnings."""

    def test_empty_string_input_no_warning(self):
        """An empty input string produces no warning (no placeholders)."""
        with patch("magic_agents.util.env_resolver.logger.warning") as mock_warn:
            result = resolve_env_string("")

        assert result == ""
        mock_warn.assert_not_called()

    def test_var_with_no_env_prefix(self):
        """A string like '{{env}}' (missing the dot and var name) does NOT match."""
        with patch("magic_agents.util.env_resolver.logger.warning") as mock_warn:
            result = resolve_env_string("{{env}}")

        assert result == "{{env}}"
        mock_warn.assert_not_called()

    def test_var_name_casing(self, monkeypatch):
        """env var names are case-sensitive — lowercase placeholder does not
        match uppercase env var, so it triggers a warning."""
        monkeypatch.setenv("MIXED_CASE_VAR", "value")

        with patch("magic_agents.util.env_resolver.logger.warning") as mock_warn:
            result = resolve_env_string("{{env.mixed_case_var}}")

        assert result == ""
        mock_warn.assert_called_once()
