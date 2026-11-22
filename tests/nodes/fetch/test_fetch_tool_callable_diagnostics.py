"""
Phase 2 / Task 2.3 -- FetchToolCallable diagnostics & body tests.

Unit tests for ``FetchToolCallable.__call__()`` non-2xx response handling,
body passthrough, diagnostic logging, body redaction, and exception logging.

All tests use mocked ``aiohttp.ClientSession`` -- no live HTTP calls.
"""
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from magic_agents.node_system.NodeFetch import (
    FetchToolCallable,
    _json_parse_mapping,
    _redact_body_preview,
)

# ---------------------------------------------------------------------------
# Helper: build a mock aiohttp session that returns a controlled response
# ---------------------------------------------------------------------------


def _make_mock_response(
    status: int,
    reason: str = "OK",
    body: str = "",
) -> MagicMock:
    """Build a mock ``aiohttp.ClientResponse``.

    The return value mimics a response used inside ``async with session.request(...) as resp``.
    """
    mock = MagicMock(spec=aiohttp.ClientResponse)
    mock.status = status
    mock.reason = reason
    mock.text = AsyncMock(return_value=body)
    return mock


def _make_mock_session(mock_response: MagicMock) -> MagicMock:
    """Build a mock ``aiohttp.ClientSession``.

    The session is an async context manager, and ``session.request(...)``
    returns another async context manager that yields ``mock_response``.
    """
    # Response context manager: async with response_cm as resp
    response_cm = MagicMock()
    response_cm.__aenter__ = AsyncMock(return_value=mock_response)
    response_cm.__aexit__ = AsyncMock(return_value=None)

    # Session itself is an async context manager
    session = MagicMock(spec=aiohttp.ClientSession)
    session.request = MagicMock(return_value=response_cm)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


def _make_mock_session_raises(exception: Exception) -> MagicMock:
    """Build a mock session whose ``request()`` raises *exception*."""
    session = MagicMock(spec=aiohttp.ClientSession)
    session.request = MagicMock(side_effect=exception)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


# ---------------------------------------------------------------------------
# Minimal FetchToolCallable factory for tests
# ---------------------------------------------------------------------------


def _make_tool(tool_name: str = "search_web", **overrides) -> FetchToolCallable:
    """Create a ``FetchToolCallable`` with sensible defaults."""
    kwargs = dict(
        url_template="https://api.example.com/search",
        method="GET",
        headers={"Accept": "application/json"},
        tool_name=tool_name,
    )
    kwargs.update(overrides)
    return FetchToolCallable(**kwargs)


# ===================================================================
# _redact_body_preview() unit tests
# ===================================================================


class TestRedactBodyPreview:
    """Direct unit tests for the ``_redact_body_preview()`` helper."""

    def test_redact_sensitive_keys_json(self):
        """Sensitive keys (api_key, token) have values replaced with ``[REDACTED]``."""
        body = json.dumps({
            "api_key": "sk-abc123",
            "token": "eyJhbGci",
            "error": "forbidden",
            "message": "Access denied",
        })
        result = _redact_body_preview(body)
        parsed = json.loads(result)
        assert parsed["api_key"] == "[REDACTED]"
        assert parsed["token"] == "[REDACTED]"
        assert parsed["error"] == "forbidden"
        assert parsed["message"] == "Access denied"

    @pytest.mark.parametrize("key", [
        "api_key", "API_KEY", "ApiKey", "api-key",
        "secret", "SECRET", "token", "TOKEN",
        "authorization", "Authorization", "auth", "Auth",
        "password", "PASSWORD", "passwd",
        "credential", "Credential",
        "access_key", "access-key", "AccessKey",
        "private_key", "private-key", "PrivateKey",
    ])
    def test_all_sensitive_key_variants(self, key):
        """All known sensitive key patterns are redacted regardless of case/hyphen."""
        body = json.dumps({key: "sensitive_value", "safe": "visible"})
        result = _redact_body_preview(body)
        parsed = json.loads(result)
        assert parsed[key] == "[REDACTED]", f"Key '{key}' should be redacted"
        assert parsed["safe"] == "visible"

    def test_non_sensitive_keys_preserved(self):
        """Non-sensitive keys like ``error``, ``message``, ``retry_after`` are NOT redacted."""
        body = json.dumps({
            "error": "Rate limit exceeded",
            "message": "Try again later",
            "retry_after": 60,
            "code": 429,
        })
        result = _redact_body_preview(body)
        parsed = json.loads(result)
        assert parsed["error"] == "Rate limit exceeded"
        assert parsed["message"] == "Try again later"
        assert parsed["retry_after"] == 60
        assert parsed["code"] == 429

    def test_non_json_body_truncation_only(self):
        """Non-JSON bodies receive only truncation, no key-redaction attempted."""
        body = "Internal Server Error: upstream timeout"
        result = _redact_body_preview(body)
        assert result == body  # unchanged (not JSON, truncation only)
        assert "[REDACTED]" not in result

    def test_empty_body(self):
        """Empty string returns empty string, no crash."""
        result = _redact_body_preview("")
        assert result == ""

    def test_json_array_body_redacts_nested_mapping_values(self):
        """Sensitive mapping keys are redacted recursively inside arrays."""
        body = json.dumps([{"api_key": "sk-abc"}, {"error": "err"}])
        result = _redact_body_preview(body)
        parsed = json.loads(result)

        assert parsed == [{"api_key": "[REDACTED]"}, {"error": "err"}]
        assert "sk-abc" not in result

    def test_body_over_max_length_truncated(self):
        """Body longer than ``max_length`` is truncated to that length."""
        body = "x" * 1000
        result = _redact_body_preview(body, max_length=500)
        assert len(result) == 500

    def test_body_over_max_length_sensitive_keys_partially_included(self):
        """Valid JSON is redacted in full before its safe preview is truncated."""
        body = '{"error":"ok",' + '"api_key":"sk-abc123",' + '"data":' + json.dumps("x" * 800) + "}"
        max_len = 500
        result = _redact_body_preview(body, max_length=max_len)

        assert "sk-abc123" not in result
        assert "[REDACTED]" in result
        assert len(result) == max_len

    def test_sensitive_keys_are_redacted_at_every_depth(self):
        """Nested mappings are redacted rather than leaking through logs."""
        body = json.dumps({
            "api_key": "sk-abc",
            "nested": {"api_key": "nested-secret", "token": "nested-token"},
        })
        result = _redact_body_preview(body)
        parsed = json.loads(result)
        assert parsed["api_key"] == "[REDACTED]"
        assert parsed["nested"]["api_key"] == "[REDACTED]"
        assert parsed["nested"]["token"] == "[REDACTED]"
        assert "nested-secret" not in result
        assert "nested-token" not in result

    def test_long_malformed_json_redacts_secret_before_truncation(self):
        """A truncated quoted value cannot make an earlier secret loggable."""
        body = (
            '{"outer":{"authorization":"Bearer malformed-secret",'
            '"payload":"' + ("x" * 1000)
        )

        result = _redact_body_preview(body, max_length=500)

        assert len(result) <= 500
        assert "malformed-secret" not in result
        assert "[REDACTED]" in result

    def test_unclosed_sensitive_value_is_fully_suppressed(self):
        """An unterminated sensitive string never reaches the log preview."""
        body = '{"api_key":"sk-unclosed-secret-' + ("x" * 1000)

        result = _redact_body_preview(body, max_length=500)

        assert "sk-unclosed-secret" not in result
        assert result == '{"api_key":"[REDACTED]"'

    def test_invalid_key_value_text_redacts_unquoted_authorization(self):
        """Non-JSON diagnostics redact header-style and assignment values."""
        body = "upstream failure\nauthorization: Bearer plaintext-secret\nretry=true"

        result = _redact_body_preview(body)

        assert "plaintext-secret" not in result
        assert "authorization: \"[REDACTED]\"" in result
        assert "retry=true" in result

    def test_max_length_custom(self):
        """``max_length`` parameter is respected for custom values."""
        body = json.dumps({"api_key": "sk-abc123", "error": "err"})
        result = _redact_body_preview(body, max_length=10)
        # Redaction happens before truncation, even if the short preview cuts the marker.
        assert len(result) <= 10
        assert "sk-abc123" not in result


# ===================================================================
# FetchToolCallable.__call__() — body passthrough & logging tests
# ===================================================================


class TestFetchToolCallableDiagnostics:
    """Tests for non-2xx body passthrough, logging, and redaction."""

    # ── Body passthrough: return value includes full unredacted body ──

    @pytest.mark.asyncio
    async def test_http_401_includes_full_body_in_return(self):
        """HTTP 401 return value includes the full response body."""
        body = '{"error":"Invalid API key"}'
        mock_resp = _make_mock_response(status=401, reason="Unauthorized", body=body)
        mock_sess = _make_mock_session(mock_resp)
        tool = _make_tool()

        with patch("aiohttp.ClientSession", return_value=mock_sess):
            result = await tool()

        expected = f"HTTP 401: Unauthorized\n\nResponse body:\n{body}"
        assert result == expected

    @pytest.mark.asyncio
    async def test_http_429_includes_full_body_in_return(self):
        """HTTP 429 return value includes the full response body."""
        body = '{"error":"Rate limit exceeded","retry_after":60}'
        mock_resp = _make_mock_response(status=429, reason="Too Many Requests", body=body)
        mock_sess = _make_mock_session(mock_resp)
        tool = _make_tool()

        with patch("aiohttp.ClientSession", return_value=mock_sess):
            result = await tool()

        expected = f"HTTP 429: Too Many Requests\n\nResponse body:\n{body}"
        assert result == expected

    @pytest.mark.asyncio
    async def test_http_500_includes_full_body_in_return(self):
        """HTTP 500 return value includes (possibly non-JSON) full body."""
        body = "upstream connection refused"
        mock_resp = _make_mock_response(status=500, reason="Internal Server Error", body=body)
        mock_sess = _make_mock_session(mock_resp)
        tool = _make_tool()

        with patch("aiohttp.ClientSession", return_value=mock_sess):
            result = await tool()

        expected = f"HTTP 500: Internal Server Error\n\nResponse body:\n{body}"
        assert result == expected

    # ── Return value is a plain string (not dict / ToolResult) ──

    @pytest.mark.asyncio
    async def test_return_value_is_plain_string(self):
        """Non-2xx return value is a plain string, not a dict or structured type."""
        body = '{"error":"Invalid API key"}'
        mock_resp = _make_mock_response(status=401, reason="Unauthorized", body=body)
        mock_sess = _make_mock_session(mock_resp)
        tool = _make_tool()

        with patch("aiohttp.ClientSession", return_value=mock_sess):
            result = await tool()

        assert isinstance(result, str)
        assert not isinstance(result, dict)

    # ── Return value body is FULL and unredacted ──

    @pytest.mark.asyncio
    async def test_return_value_body_is_unredacted(self):
        """The body in the return value is the FULL unredacted text."""
        body = '{"api_key":"sk-abc123","token":"eyJhbGci","error":"forbidden"}'
        mock_resp = _make_mock_response(status=403, reason="Forbidden", body=body)
        mock_sess = _make_mock_session(mock_resp)
        tool = _make_tool()

        with patch("aiohttp.ClientSession", return_value=mock_sess):
            result = await tool()

        # The return value must contain the FULL body with sensitive values visible
        assert "sk-abc123" in result
        assert "eyJhbGci" in result
        assert "[REDACTED]" not in result

    # ── logger.warning is called for non-2xx responses ──

    @pytest.mark.asyncio
    async def test_logger_warning_called_for_http_401(self, caplog):
        """``logger.warning`` is called with tool name, status, and reason for HTTP 401."""
        caplog.set_level(logging.WARNING, logger="magic_agents.node_system.NodeFetch")
        body = '{"error":"Invalid API key"}'
        mock_resp = _make_mock_response(status=401, reason="Unauthorized", body=body)
        mock_sess = _make_mock_session(mock_resp)
        tool = _make_tool(tool_name="search_web")

        with patch("aiohttp.ClientSession", return_value=mock_sess):
            await tool()

        assert len(caplog.records) >= 1
        log_msg = caplog.records[0].getMessage()
        assert "search_web" in log_msg
        assert "401" in log_msg
        assert "Unauthorized" in log_msg

    @pytest.mark.asyncio
    async def test_logger_warning_called_for_http_500(self, caplog):
        """``logger.warning`` is called for HTTP 500 with correct details."""
        caplog.set_level(logging.WARNING, logger="magic_agents.node_system.NodeFetch")
        body = "upstream timeout"
        mock_resp = _make_mock_response(status=500, reason="Internal Server Error", body=body)
        mock_sess = _make_mock_session(mock_resp)
        tool = _make_tool(tool_name="scrape_url")

        with patch("aiohttp.ClientSession", return_value=mock_sess):
            await tool()

        assert len(caplog.records) >= 1
        log_msg = caplog.records[0].getMessage()
        assert "scrape_url" in log_msg
        assert "500" in log_msg
        assert "Internal Server Error" in log_msg

    # ── logger.warning NOT called for 2xx responses ──

    @pytest.mark.asyncio
    async def test_http_200_does_not_log_warning(self, caplog):
        """HTTP 200 response does NOT trigger ``logger.warning``."""
        caplog.set_level(logging.WARNING, logger="magic_agents.node_system.NodeFetch")
        body = '{"results": []}'
        mock_resp = _make_mock_response(status=200, reason="OK", body=body)
        mock_sess = _make_mock_session(mock_resp)
        tool = _make_tool()

        with patch("aiohttp.ClientSession", return_value=mock_sess):
            await tool()

        # No WARNING-level log for 2xx
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) == 0

    @pytest.mark.asyncio
    async def test_http_204_does_not_log_warning(self, caplog):
        """HTTP 204 (No Content) does NOT trigger ``logger.warning``."""
        caplog.set_level(logging.WARNING, logger="magic_agents.node_system.NodeFetch")
        body = ""
        mock_resp = _make_mock_response(status=204, reason="No Content", body=body)
        mock_sess = _make_mock_session(mock_resp)
        tool = _make_tool()

        with patch("aiohttp.ClientSession", return_value=mock_sess):
            await tool()

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) == 0

    # ── Log body preview: sensitive keys redacted, non-sensitive visible ──

    @pytest.mark.asyncio
    async def test_log_body_preview_redacts_sensitive_keys(self, caplog):
        """Log body preview redacts ``api_key`` and ``token`` values."""
        caplog.set_level(logging.WARNING, logger="magic_agents.node_system.NodeFetch")
        body = '{"error":"forbidden","api_key":"sk-abc123","token":"eyJhbGci","message":"Access denied"}'
        mock_resp = _make_mock_response(status=403, reason="Forbidden", body=body)
        mock_sess = _make_mock_session(mock_resp)
        tool = _make_tool()

        with patch("aiohttp.ClientSession", return_value=mock_sess):
            await tool()

        assert len(caplog.records) >= 1
        log_msg = caplog.records[0].getMessage()
        # Sensitive values must NOT appear in log
        assert "sk-abc123" not in log_msg
        assert "eyJhbGci" not in log_msg
        # Log should show redaction markers
        assert "[REDACTED]" in log_msg

    @pytest.mark.asyncio
    async def test_log_body_preview_preserves_non_sensitive(self, caplog):
        """Log body preview preserves non-sensitive keys like ``error`` and ``message``."""
        caplog.set_level(logging.WARNING, logger="magic_agents.node_system.NodeFetch")
        body = '{"error":"forbidden","message":"Access denied"}'
        mock_resp = _make_mock_response(status=403, reason="Forbidden", body=body)
        mock_sess = _make_mock_session(mock_resp)
        tool = _make_tool()

        with patch("aiohttp.ClientSession", return_value=mock_sess):
            await tool()

        assert len(caplog.records) >= 1
        log_msg = caplog.records[0].getMessage()
        assert "forbidden" in log_msg
        assert "Access denied" in log_msg

    @pytest.mark.asyncio
    async def test_log_body_preview_truncation(self, caplog):
        """Log body preview is truncated — the full body appears in return but not in log."""
        caplog.set_level(logging.WARNING, logger="magic_agents.node_system.NodeFetch")
        # 600-char body — the log preview is truncated to 500, but the return has all 600
        long_payload = "x" * 600
        mock_resp = _make_mock_response(status=500, reason="Error", body=long_payload)
        mock_sess = _make_mock_session(mock_resp)
        tool = _make_tool()

        with patch("aiohttp.ClientSession", return_value=mock_sess):
            result = await tool()

        # The return value has the full body
        assert long_payload in result

        # The log preview has at most 500 chars of the body
        assert len(caplog.records) >= 1
        log_msg = caplog.records[0].getMessage()
        # The "Response body:" section in the log should not have >500 consecutive 'x'
        body_section = log_msg[log_msg.index("Response body:"):]
        # Count consecutive x's — should not exceed 500
        chunk = "x" * 501
        assert chunk not in body_section

    # ── No secrets / headers in log output ──

    @pytest.mark.asyncio
    async def test_no_headers_in_log(self, caplog):
        """No request headers appear in the log message."""
        caplog.set_level(logging.WARNING, logger="magic_agents.node_system.NodeFetch")
        body = '{"error":"forbidden"}'
        mock_resp = _make_mock_response(status=403, reason="Forbidden", body=body)
        mock_sess = _make_mock_session(mock_resp)
        # Include an API-key-like header in the tool config
        tool = _make_tool(headers={"X-API-KEY": "sk-super-secret-key"})

        with patch("aiohttp.ClientSession", return_value=mock_sess):
            await tool()

        assert len(caplog.records) >= 1
        log_msg = caplog.records[0].getMessage()
        # No header content in the log
        assert "X-API-KEY" not in log_msg
        assert "sk-super-secret-key" not in log_msg
        assert "Accept" not in log_msg

    # ── Tool name in log matches `FetchToolCallable._tool_name` ──

    @pytest.mark.asyncio
    async def test_log_contains_tool_name(self, caplog):
        """Tool name in log message matches ``FetchToolCallable._tool_name``."""
        caplog.set_level(logging.WARNING, logger="magic_agents.node_system.NodeFetch")
        body = '{"error":"err"}'
        mock_resp = _make_mock_response(status=400, reason="Bad Request", body=body)
        mock_sess = _make_mock_session(mock_resp)
        tool = _make_tool(tool_name="my_custom_tool")

        with patch("aiohttp.ClientSession", return_value=mock_sess):
            await tool()

        assert len(caplog.records) >= 1
        log_msg = caplog.records[0].getMessage()
        assert "my_custom_tool" in log_msg

    # ── Exception paths ──

    @pytest.mark.asyncio
    async def test_aiohttp_client_error_logs_warning(self, caplog):
        """``aiohttp.ClientError`` triggers ``logger.warning`` with tool name and message."""
        caplog.set_level(logging.WARNING, logger="magic_agents.node_system.NodeFetch")
        exc = aiohttp.ClientError("Connection refused")
        mock_sess = _make_mock_session_raises(exc)
        tool = _make_tool(tool_name="search_web")

        with patch("aiohttp.ClientSession", return_value=mock_sess):
            result = await tool()

        assert len(caplog.records) >= 1
        log_msg = caplog.records[0].getMessage()
        assert "search_web" in log_msg
        assert "Connection refused" in log_msg
        # Return value is unchanged from existing behavior
        expected = json.dumps({"error": "Network error: Connection refused"})
        assert result == expected

    @pytest.mark.asyncio
    async def test_unexpected_exception_logs_warning(self, caplog):
        """Unexpected ``Exception`` triggers ``logger.warning`` with tool name and message."""
        caplog.set_level(logging.WARNING, logger="magic_agents.node_system.NodeFetch")
        exc = ValueError("Something went wrong")
        mock_sess = _make_mock_session_raises(exc)
        tool = _make_tool(tool_name="scrape_url")

        with patch("aiohttp.ClientSession", return_value=mock_sess):
            result = await tool()

        assert len(caplog.records) >= 1
        log_msg = caplog.records[0].getMessage()
        assert "scrape_url" in log_msg
        assert "ValueError" in log_msg
        assert "Something went wrong" in log_msg
        # Return value is unchanged
        expected = json.dumps({"error": "Unexpected error: Something went wrong"})
        assert result == expected

    @pytest.mark.asyncio
    async def test_aiohttp_client_response_error_logs_warning(self, caplog):
        """``aiohttp.ClientResponseError`` triggers ``logger.warning``."""
        caplog.set_level(logging.WARNING, logger="magic_agents.node_system.NodeFetch")
        exc = aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=MagicMock(),
            status=429,
            message="Too Many Requests",
        )
        mock_sess = _make_mock_session_raises(exc)
        tool = _make_tool(tool_name="search_web")

        with patch("aiohttp.ClientSession", return_value=mock_sess):
            result = await tool()

        assert len(caplog.records) >= 1
        log_msg = caplog.records[0].getMessage()
        assert "search_web" in log_msg
        assert "429" in log_msg
        assert "Too Many Requests" in log_msg
        # Return value for ClientResponseError
        assert result == "HTTP 429: Too Many Requests"

    # ── Success path (HTTP 200) returns parsed JSON, no warning ──

    @pytest.mark.asyncio
    async def test_http_200_returns_parsed_json(self):
        """HTTP 200 returns ``json.dumps(json.loads(body))`` -- no regression."""
        body = '{"results": [{"title": "hello"}]}'
        mock_resp = _make_mock_response(status=200, reason="OK", body=body)
        mock_sess = _make_mock_session(mock_resp)
        tool = _make_tool()

        with patch("aiohttp.ClientSession", return_value=mock_sess):
            result = await tool()

        # Should be the re-serialized JSON (consistent with existing behavior)
        parsed = json.loads(result)
        assert parsed == {"results": [{"title": "hello"}]}

    @pytest.mark.asyncio
    async def test_http_200_non_json_body_passthrough(self):
        """HTTP 200 with non-JSON body returns raw body (unchanged)."""
        body = "plain text response"
        mock_resp = _make_mock_response(status=200, reason="OK", body=body)
        mock_sess = _make_mock_session(mock_resp)
        tool = _make_tool()

        with patch("aiohttp.ClientSession", return_value=mock_sess):
            result = await tool()

        assert result == body

    # ── No secrets leak in exception paths ──

    @pytest.mark.asyncio
    async def test_exception_log_no_secrets(self, caplog):
        """Exception log does not contain request headers or raw args."""
        caplog.set_level(logging.WARNING, logger="magic_agents.node_system.NodeFetch")
        exc = aiohttp.ClientError("timeout")
        mock_sess = _make_mock_session_raises(exc)
        tool = _make_tool(
            tool_name="search_web",
            headers={"X-API-KEY": "sk-leaked-key"},
        )

        with patch("aiohttp.ClientSession", return_value=mock_sess):
            await tool()

        assert len(caplog.records) >= 1
        log_msg = caplog.records[0].getMessage()
        # No header content in exception log
        assert "X-API-KEY" not in log_msg
        assert "sk-leaked-key" not in log_msg
        # Tool name should be present though
        assert "search_web" in log_msg


# ===================================================================
# _json_parse_mapping + FetchToolCallable input normalization tests
# ===================================================================


class TestFetchToolCallableInputNormalization:
    """Tests for _json_parse_mapping and input normalization.

    Covers headers/params arriving as JSON strings (as UI/editor payloads
    often do), invalid strings, empty strings, and backward compatibility.
    """

    # ── _json_parse_mapping unit tests ──

    def test_json_parse_mapping_dict(self):
        """_json_parse_mapping with dict returns a copy."""
        original = {"a": 1}
        result = _json_parse_mapping(original, "headers", "tool")
        assert result == original
        assert result is not original

    def test_json_parse_mapping_valid_json_string(self):
        """_json_parse_mapping with valid JSON object string returns parsed dict."""
        result = _json_parse_mapping(
            '{"Accept": "application/json"}', "headers", "tool",
        )
        assert result == {"Accept": "application/json"}

    def test_json_parse_mapping_empty_string(self):
        """_json_parse_mapping with empty string returns {}."""
        result = _json_parse_mapping("", "headers", "tool")
        assert result == {}

    def test_json_parse_mapping_invalid_string_logs_warning(self, caplog):
        """_json_parse_mapping with invalid JSON string logs warning."""
        caplog.set_level(logging.WARNING, logger="magic_agents.node_system.NodeFetch")
        result = _json_parse_mapping("not json", "headers", "test_tool")
        assert result == {}
        assert len(caplog.records) >= 1
        msg = caplog.records[0].getMessage()
        assert "test_tool" in msg
        assert "headers" in msg
        assert "invalid JSON" in msg

    def test_json_parse_mapping_none(self):
        """_json_parse_mapping with None returns {}."""
        result = _json_parse_mapping(None, "headers", "tool")
        assert result == {}

    def test_json_parse_mapping_none_with_allow_none(self):
        """_json_parse_mapping with None and allow_none=True returns None."""
        result = _json_parse_mapping(None, "params", "tool", allow_none=True)
        assert result is None

    def test_json_parse_mapping_non_dict_json_logs_warning(self, caplog):
        """_json_parse_mapping with JSON array logs warning and returns {}."""
        caplog.set_level(logging.WARNING, logger="magic_agents.node_system.NodeFetch")
        result = _json_parse_mapping("[1, 2, 3]", "headers", "tool")
        assert result == {}
        assert len(caplog.records) >= 1
        assert "not a dict" in caplog.records[0].getMessage()

    def test_json_parse_mapping_whitespace_string(self):
        """_json_parse_mapping with whitespace-only string returns {}."""
        result = _json_parse_mapping("   ", "headers", "tool")
        assert result == {}

    # ── FetchToolCallable integration tests ──

    @pytest.mark.asyncio
    async def test_headers_json_string_reaches_http_with_parsed_headers(self):
        """headers as JSON string is parsed and reaches aiohttp with correct dict."""
        mock_resp = _make_mock_response(status=200, reason="OK", body='{"ok":true}')
        mock_sess = _make_mock_session(mock_resp)
        tool = _make_tool(headers='{"Accept": "application/json", "X-Custom": "val"}')

        with patch("aiohttp.ClientSession", return_value=mock_sess):
            await tool()

        call_kwargs = mock_sess.request.call_args[1]
        assert call_kwargs["headers"] == {
            "Accept": "application/json",
            "X-Custom": "val",
        }

    @pytest.mark.asyncio
    async def test_headers_invalid_json_uses_empty_and_logs_warning(self, caplog):
        """headers as invalid JSON string falls back to {} and logs a warning."""
        caplog.set_level(logging.WARNING, logger="magic_agents.node_system.NodeFetch")
        mock_resp = _make_mock_response(status=200, reason="OK", body='{"ok":true}')
        mock_sess = _make_mock_session(mock_resp)
        tool = _make_tool(headers="not valid json at all")

        with patch("aiohttp.ClientSession", return_value=mock_sess):
            await tool()

        call_kwargs = mock_sess.request.call_args[1]
        assert call_kwargs["headers"] == {}
        assert len(caplog.records) >= 1
        assert "invalid JSON" in caplog.records[0].getMessage()

    @pytest.mark.asyncio
    async def test_headers_empty_string_uses_empty(self):
        """headers as empty string falls back to empty dict."""
        mock_resp = _make_mock_response(status=200, reason="OK", body='{"ok":true}')
        mock_sess = _make_mock_session(mock_resp)
        tool = _make_tool(headers="")

        with patch("aiohttp.ClientSession", return_value=mock_sess):
            await tool()

        call_kwargs = mock_sess.request.call_args[1]
        assert call_kwargs["headers"] == {}

    @pytest.mark.asyncio
    async def test_headers_dict_still_works(self):
        """headers as dict works unchanged (backward compat)."""
        mock_resp = _make_mock_response(status=200, reason="OK", body='{"ok":true}')
        mock_sess = _make_mock_session(mock_resp)
        expected = {"Accept": "application/json", "X-Key": "value"}
        tool = _make_tool(headers=expected)

        with patch("aiohttp.ClientSession", return_value=mock_sess):
            await tool()

        call_kwargs = mock_sess.request.call_args[1]
        assert call_kwargs["headers"] == expected

    @pytest.mark.asyncio
    async def test_headers_none_uses_empty(self):
        """headers as None uses empty dict (backward compat)."""
        mock_resp = _make_mock_response(status=200, reason="OK", body='{"ok":true}')
        mock_sess = _make_mock_session(mock_resp)
        tool = _make_tool(headers=None)

        with patch("aiohttp.ClientSession", return_value=mock_sess):
            await tool()

        call_kwargs = mock_sess.request.call_args[1]
        assert call_kwargs["headers"] == {}

    @pytest.mark.asyncio
    async def test_params_json_string_parsed(self):
        """params as JSON string is parsed and sent to aiohttp."""
        mock_resp = _make_mock_response(status=200, reason="OK", body='{"ok":true}')
        mock_sess = _make_mock_session(mock_resp)
        tool = _make_tool(params='{"q": "hello", "num": 10}')

        with patch("aiohttp.ClientSession", return_value=mock_sess):
            await tool()

        call_kwargs = mock_sess.request.call_args[1]
        assert call_kwargs["params"] == {"q": "hello", "num": 10}

    @pytest.mark.asyncio
    async def test_params_valid_json_with_jinja2(self):
        """params JSON string with Jinja2 templates renders correctly."""
        mock_resp = _make_mock_response(status=200, reason="OK", body='{"ok":true}')
        mock_sess = _make_mock_session(mock_resp)
        tool = _make_tool(params='{"q": "{{query}}", "num": 10}')

        with patch("aiohttp.ClientSession", return_value=mock_sess):
            await tool(query="hello")

        call_kwargs = mock_sess.request.call_args[1]
        assert call_kwargs["params"] == {"q": "hello", "num": 10}

    @pytest.mark.asyncio
    async def test_json_data_string_template_still_works(self):
        """json_data as Jinja2 string template renders and is sent as json body."""
        mock_resp = _make_mock_response(status=200, reason="OK", body='{"ok":true}')
        mock_sess = _make_mock_session(mock_resp)
        tool = _make_tool(
            method="POST",
            json_data='{"query": "{{keyword}}", "num": 10}',
        )

        with patch("aiohttp.ClientSession", return_value=mock_sess):
            await tool(keyword="hello")

        call_kwargs = mock_sess.request.call_args[1]
        assert call_kwargs["json"] == {"query": "hello", "num": 10}

    @pytest.mark.asyncio
    async def test_headers_json_string_with_jinja2_env(self):
        """headers JSON string with Jinja2 env templates renders correctly."""
        mock_resp = _make_mock_response(status=200, reason="OK", body='{"ok":true}')
        mock_sess = _make_mock_session(mock_resp)
        tool = _make_tool(headers='{"X-API-KEY": "{{key}}", "Accept": "application/json"}')

        with patch("aiohttp.ClientSession", return_value=mock_sess):
            await tool(key="sk-test-key")

        call_kwargs = mock_sess.request.call_args[1]
        assert call_kwargs["headers"]["X-API-KEY"] == "sk-test-key"
        assert call_kwargs["headers"]["Accept"] == "application/json"
