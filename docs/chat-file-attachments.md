# Chat file attachments

The UI sends each document as an attachment descriptor, separate from the user prompt:

```json
{
  "name": "notes.md",
  "mime_type": "text/markdown",
  "data": "data:text/markdown;base64,IyBOb3Rlcw=="
}
```

Pass descriptors with `magic_agents.agt_flow.build(..., files=[...])`. The request value overrides UserInput's configured files; `None` preserves graph defaults and `[]` clears them. Connect UserInput's `handle_user_files` output to Chat's `handle_user_files` input. Chat accepts file-only turns and files together with images. Each turn supports up to four documents: UTF-8 `.md` and `.txt` files up to 1 MiB each, and `.pdf` files up to 10 MiB each.

NodeChat decodes the attachments and injects their contents into user messages immediately before the associated user prompt. Stored history can keep the original descriptors in each user message's `files` field. `await node.prepare_messages(messages)` handles the same conversion for direct chats without a graph, without modifying persisted messages. Legacy extracted `[text, image]` pairs and `{text: ...}` descriptors still work.

## PDF extraction and Redis

The Magic package installs `pdf-inspector>=1.25.2` and `redis>=5.0`. PDFs are parsed on the Chat node using `pdf_inspector.process_pdf(path)`. Only when `result.markdown is None` does the node call `pdf_inspector.process_pdf_with_ocr(path)`. Native extraction runs in a worker thread and its secure temporary file is deleted after processing.

Configure the Chat node with a server environment variable reference:

```json
{
  "type": "chat",
  "id": "chat",
  "data": {
    "file_cache_redis": {
      "url_env": "REDIS_URL",
      "ttl_seconds": 86400
    }
  }
}
```

Set `REDIS_URL=redis://localhost:6379/15` in the runtime environment. Keep connection credentials in that server environment, rather than saved agent graphs. Headless callers can instead provide `{url: ...}` or Redis `host`, `port`, `db`, and other connection settings. An injected `file_cache` constructor dependency takes precedence; its async `get`/`set` methods must follow the Redis interface, and the caller owns its lifecycle.

The cache key includes a schema version and SHA-256 digest of the PDF bytes. Redis stores parsed Markdown, PDF classification, page count, OCR use, and routed OCR page numbers with the configured TTL. Renaming a file reuses its parsed contents; changing the bytes creates a new entry. Invalid cache entries are reparsed and replaced. Missing configuration, a missing environment variable, or a failed Redis connection generates a `FileCacheWarning` and parsing continues. The node's debug state exposes `file_cache_configured` and `file_cache_warnings`. Owned Redis connections are closed after preparation; `await node.aclose()` is safe to call again.

## OCR runtime setup

Scanned PDFs require separate shared libraries and OCR models. The [upstream setup guide](https://github.com/firecrawl/pdf-inspector/blob/main/docs/ocr-runtime.md) specifies Firecrawl PDFium `native-v7988`, ONNX Runtime `1.27.0`, and PP-OCRv6 Small revision `oar-ocr-v0.7.0`.

Extract the appropriate platform archives and point `PDFIUM_LIB_PATH` and `ORT_DYLIB_PATH` at their shared libraries. For Linux x64 the archives are `firecrawl-pdfium-linux-x64.tgz` and `onnxruntime-linux-x64-1.27.0.tgz`:

```bash
export PDFIUM_LIB_PATH=/opt/pdfium/lib/libpdfium.so
export ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so
export PDF_INSPECTOR_MODEL_CACHE=/var/cache/pdf-inspector
```

The first OCR request downloads and verifies the pinned model files into that cache. Ensure the runtime can write there and can fetch the models. Clean text PDFs use native extraction without loading those libraries or downloading models. Missing libraries, model acquisition errors, or empty parsed text produce an attachment validation error instead of sending an incomplete document to the provider.
