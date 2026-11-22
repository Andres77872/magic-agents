"""Debug redaction honours configured keys and leaves usage counters readable."""
import pytest

from magic_agents.debug.config import DebugConfig
from magic_agents.debug.context import DebugContext
from magic_agents.debug.transform import RedactTransformer


@pytest.mark.parametrize('key', ['prompt_tokens', 'completion_tokens', 'max_tokens', 'maxTokens',
                                 'reasoning_tokens', 'author'])
def test_usage_counters_and_ordinary_words_are_not_redacted(key):
    assert not RedactTransformer()._is_sensitive_key(key)


@pytest.mark.parametrize('key', ['token', 'access_token', 'accessToken', 'refresh_tokens', 'id_token',
                                 'x-auth-token', 'auth', 'auth_header', 'oauth', 'Authorization',
                                 'X-Api-Key', 'user_password', 'client_secret', 'pwd'])
def test_secret_keys_are_still_redacted(key):
    assert RedactTransformer()._is_sensitive_key(key)


def test_debug_context_applies_configured_additional_keys():
    config = DebugConfig.from_dict({'redact_sensitive': True, 'additional_redact_keys': ['tenant_pin']})
    pipeline = DebugContext(config=config)._pipeline
    redactor = next(t for t in pipeline._transformers if isinstance(t, RedactTransformer))
    assert redactor._is_sensitive_key('tenant_pin')
