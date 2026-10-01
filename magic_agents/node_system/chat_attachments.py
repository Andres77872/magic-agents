"""Decode real chat file attachments and cache PDF extraction results."""
from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import importlib
import inspect
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from magic_agents.util.env_resolver import resolve_env_placeholders

logger = logging.getLogger(__name__)

MAX_CHAT_FILES = 4
MAX_TEXT_FILE_BYTES = 1024 * 1024
MAX_PDF_FILE_BYTES = 10 * 1024 * 1024
CACHE_SCHEMA_VERSION = 1
MIME_BY_EXTENSION = {'.md': 'text/markdown', '.txt': 'text/plain', '.pdf': 'application/pdf'}


class ChatFileError(ValueError):
    """A file could not be decoded or parsed for the chat input."""


class ChatAttachmentProcessor:
    """NodeChat's file boundary; Redis is optional and never changes parsing."""

    def __init__(self, *, redis_config: dict[str, Any] | None = None,
                 cache: Any = None, ttl_seconds: int = 86400) -> None:
        self.redis_config = redis_config
        self.cache = cache
        self.ttl_seconds = ttl_seconds
        self._owned_cache = None
        self.warnings: list[str] = []

    def _warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)
            logger.warning(message)

    async def _get_cache(self):
        if self.cache is not None:
            return self.cache
        if self._owned_cache is not None:
            return self._owned_cache
        if not self.redis_config:
            self._warn('PDF attachment cache is unavailable: configure a Redis database on the Chat node.')
            return None
        try:
            config = resolve_env_placeholders(dict(self.redis_config))
            ttl = config.pop('ttl_seconds', self.ttl_seconds)
            if not isinstance(ttl, int) or isinstance(ttl, bool) or ttl < 1:
                raise ValueError('cache ttl_seconds must be a positive integer')
            self.ttl_seconds = ttl
            env_name = config.pop('url_env', None)
            if env_name is not None:
                if not isinstance(env_name, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', env_name):
                    raise ValueError('url_env must name a server environment variable')
                url = os.environ.get(env_name)
                if not url:
                    self._warn(f'PDF attachment cache is unavailable: Redis URL environment variable {env_name} is not set.')
                    return None
                config['url'] = url
            redis_async = importlib.import_module('redis.asyncio')
            url = config.pop('url', None)
            config.setdefault('socket_connect_timeout', 2)
            config.setdefault('socket_timeout', 2)
            if url:
                if not isinstance(url, str) or not url.startswith(('redis://', 'rediss://')):
                    raise ValueError('Redis URL must start with redis:// or rediss://')
                self._owned_cache = redis_async.Redis.from_url(url, **config)
            elif isinstance(config.get('host'), str) and config['host'].strip():
                self._owned_cache = redis_async.Redis(**config)
            else:
                raise ValueError('Redis configuration requires url_env, url or host')
            return self._owned_cache
        except Exception as exc:
            # Redis library exceptions can embed credentials in their text.
            self._warn(f'PDF attachment cache is unavailable: check the Chat node Redis setup ({type(exc).__name__}).')
            return None

    @staticmethod
    async def _cache_call(method, *args, **kwargs):
        if inspect.iscoroutinefunction(method):
            return await method(*args, **kwargs)
        result = await asyncio.to_thread(method, *args, **kwargs)
        return await result if inspect.isawaitable(result) else result

    @staticmethod
    def decode_file(value: dict[str, Any]) -> tuple[str, str, bytes]:
        if not isinstance(value, dict):
            raise ChatFileError('File attachments must be descriptor objects.')
        name = value.get('name')
        mime = value.get('mime_type')
        data = value.get('data')
        if not isinstance(name, str) or not name.strip():
            raise ChatFileError('File attachments require a filename.')
        # Keep the filename as a label, never as a local path.
        name = name.replace('\\', '/').rsplit('/', 1)[-1]
        if len(name) > 255 or any(ord(char) < 32 for char in name):
            raise ChatFileError('File attachment filename is invalid.')
        expected = MIME_BY_EXTENSION.get(Path(name).suffix.lower())
        if expected is None:
            raise ChatFileError('Chat files must have .md, .txt or .pdf extensions.')
        if mime != expected:
            raise ChatFileError(f'File {name} has an unsupported MIME type.')
        limit = MAX_PDF_FILE_BYTES if expected == 'application/pdf' else MAX_TEXT_FILE_BYTES
        if not isinstance(data, str) or len(data) > 4 * ((limit + 2) // 3) + 128:
            raise ChatFileError(f'File {name} exceeds its attachment size limit.')
        match = re.fullmatch(r'data:([^;,]+);base64,([A-Za-z0-9+/=]*)', data)
        if match is None or match.group(1).lower() != expected:
            raise ChatFileError(f'File {name} requires a base64 data URI with its MIME type.')
        try:
            payload = base64.b64decode(match.group(2), validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ChatFileError(f'File {name} contains invalid base64 data.') from exc
        if not payload or len(payload) > limit:
            raise ChatFileError(f'File {name} is empty or exceeds its attachment size limit.')
        if expected == 'application/pdf' and not payload.startswith(b'%PDF-'):
            raise ChatFileError(f'File {name} does not contain a PDF document.')
        return name, expected, payload

    @staticmethod
    def _parse_pdf(payload: bytes) -> dict[str, Any]:
        try:
            pdf_inspector = importlib.import_module('pdf_inspector')
        except ImportError as exc:
            raise ChatFileError('PDF processing requires the pdf-inspector package in the Magic runtime.') from exc
        # Public native/OCR path APIs match pdf-inspector's documented contract.
        # The secure temporary directory is removed even when the parser fails.
        try:
            with tempfile.TemporaryDirectory(prefix='magic-chat-pdf-') as directory:
                path = Path(directory) / 'attachment.pdf'
                path.write_bytes(payload)
                native = pdf_inspector.process_pdf(str(path))
                markdown = native.markdown
                routed: list[int] = []
                used_ocr = markdown is None
                page_count = getattr(native, 'page_count', None)
                if used_ocr:
                    ocr = pdf_inspector.process_pdf_with_ocr(str(path))
                    markdown = ocr.markdown
                    routed = list(getattr(ocr, 'pages_routed_to_ocr', []))
                    page_count = getattr(ocr, 'page_count', page_count)
                if not isinstance(markdown, str) or not markdown.strip():
                    raise ChatFileError('PDF processing did not return any document text.')
                return {'markdown': markdown, 'pdf_type': getattr(native, 'pdf_type', None),
                        'page_count': page_count, 'used_ocr': used_ocr,
                        'pages_routed_to_ocr': routed}
        except ChatFileError:
            raise
        except Exception as exc:
            raise ChatFileError(
                'PDF parsing failed. Scanned PDFs require the pdf-inspector PDFium, '
                f'ONNX Runtime and OCR model setup ({type(exc).__name__}).'
            ) from exc

    async def _pdf_markdown(self, payload: bytes) -> str:
        digest = hashlib.sha256(payload).hexdigest()
        key = f'magic_agents:chat:pdf:v{CACHE_SCHEMA_VERSION}:{digest}'
        cache = await self._get_cache()
        if cache is not None:
            try:
                raw = await self._cache_call(cache.get, key)
                if raw is not None:
                    cached = json.loads(raw)
                    if (isinstance(cached, dict) and cached.get('version') == CACHE_SCHEMA_VERSION
                            and cached.get('sha256') == digest
                            and isinstance(cached.get('markdown'), str) and cached['markdown'].strip()):
                        return cached['markdown']
            except Exception as exc:
                self._warn(f'PDF attachment cache could not be read; check Redis availability ({type(exc).__name__}).')
        result = await asyncio.to_thread(self._parse_pdf, payload)
        if cache is not None:
            try:
                await self._cache_call(cache.set, key, json.dumps(
                    {'version': CACHE_SCHEMA_VERSION, 'sha256': digest, **result}, ensure_ascii=False,
                ), ex=self.ttl_seconds)
            except Exception as exc:
                self._warn(f'Parsed PDF could not be cached; check Redis availability ({type(exc).__name__}).')
        return result['markdown']

    async def parse_file(self, value: dict[str, Any]) -> str:
        name, mime, payload = self.decode_file(value)
        if mime == 'application/pdf':
            text = await self._pdf_markdown(payload)
        else:
            try:
                text = payload.decode('utf-8-sig')
            except UnicodeDecodeError as exc:
                raise ChatFileError(f'File {name} must contain UTF-8 text.') from exc
            if '\x00' in text or not text.strip():
                raise ChatFileError(f'File {name} must contain non-empty UTF-8 text.')
        return f'Attached file: {name}\n\n{text}'

    async def aclose(self) -> None:
        cache, self._owned_cache = self._owned_cache, None
        if cache is not None:
            close = getattr(cache, 'aclose', None) or getattr(cache, 'close', None)
            if close is not None:
                try:
                    await self._cache_call(close)
                except Exception:
                    logger.warning('Could not close the Chat node Redis cache connection.')
