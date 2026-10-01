"""Real files stay descriptors until NodeChat assembles provider messages."""
import base64
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from magic_agents.agt_flow import build, run_agent
from magic_agents.models.factory.Nodes.ChatNodeModel import ChatNodeModel
from magic_agents.node_system.NodeChat import NodeChat
from magic_agents.node_system.chat_attachments import ChatAttachmentProcessor, ChatFileError


def attachment(name='notes.md', payload=b'# Notes\nHello', mime='text/markdown'):
    return {'name': name, 'mime_type': mime,
            'data': f'data:{mime};base64,' + base64.b64encode(payload).decode('ascii')}


def extract(outputs):
    return next(item['content']['content'] for item in outputs if item['type'] == 'handle_chat_output')


def graph_definition(files=None):
    return {'type': 'graph', 'contract_config': {'mode': 'strict'}, 'nodes': [
        {'id': 'input', 'type': 'user_input', 'data': {'files': files or []}},
        {'id': 'chat', 'type': 'chat'}, {'id': 'end', 'type': 'end'},
    ], 'edges': [
        {'id': 'message', 'source': 'input', 'target': 'chat',
         'sourceHandle': 'handle_user_message', 'targetHandle': 'handle_user_message'},
        {'id': 'files', 'source': 'input', 'target': 'chat',
         'sourceHandle': 'handle_user_files', 'targetHandle': 'handle_user_files'},
        {'id': 'output', 'source': 'chat', 'target': 'end',
         'sourceHandle': 'handle_chat_output', 'targetHandle': 'handle_flow_input'},
    ]}


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.writes = []
    async def get(self, key):
        return self.values.get(key)
    async def set(self, key, value, ex):
        self.values[key] = value
        self.writes.append((key, json.loads(value), ex))


def parser_mock(monkeypatch, *, markdown='PDF text', ocr_markdown='OCR text'):
    paths = []
    def parse(path):
        paths.append(path)
        assert Path(path).read_bytes().startswith(b'%PDF-')
        return SimpleNamespace(markdown=markdown, pdf_type='scanned' if markdown is None else 'text_based', page_count=1)
    parser = SimpleNamespace(process_pdf=Mock(side_effect=parse), process_pdf_with_ocr=Mock(
        return_value=SimpleNamespace(markdown=ocr_markdown, pages_routed_to_ocr=[1], page_count=1)))
    monkeypatch.setitem(sys.modules, 'pdf_inspector', parser)
    return parser, paths


@pytest.mark.asyncio
async def test_real_graph_request_files_override_defaults_and_file_only_send():
    original = graph_definition([attachment(payload=b'Default')])
    before = copy.deepcopy(original)
    request = [attachment('question.txt', b'actual request', 'text/plain')]
    graph = build(original, message='', files=request)
    assert graph.nodes['input'].files == request
    assert original == before
    request[0]['name'] = 'changed.txt'
    assert graph.nodes['input'].files[0]['name'] == 'question.txt'
    assert not graph.contract_report.has_errors()
    async for _ in run_agent(graph):
        pass
    assert graph.nodes['chat'].chat.messages == [
        {'role': 'user', 'content': 'Attached file: question.txt\n\nactual request'}]
    assert graph.nodes['end'].response is not None
    assert build(original, message='x', files=[]).nodes['input'].files == []
    assert build(original, message='x').nodes['input'].files == before['nodes'][0]['data']['files']


@pytest.mark.asyncio
async def test_history_files_reparsed_by_chat_without_mutating_persisted_history():
    history = [{'role': 'user', 'content': 'old prompt', 'files': [attachment(payload=b'history text')]},
               {'role': 'assistant', 'content': 'old answer'}]
    before = copy.deepcopy(history)
    node = NodeChat(ChatNodeModel(history_messages=history))
    node.inputs[node.INPUT_HANDLER_USER_MESSAGE] = 'new prompt'
    output = extract([item async for item in node.process(None)])
    assert output.messages == [
        {'role': 'user', 'content': 'Attached file: notes.md\n\nhistory text'},
        {'role': 'user', 'content': 'old prompt'},
        {'role': 'assistant', 'content': 'old answer'},
        {'role': 'user', 'content': 'new prompt'},
    ]
    assert history == before
    assert all('files' not in message for message in output.messages)


@pytest.mark.asyncio
async def test_mixed_files_images_and_images_only():
    node = NodeChat(ChatNodeModel())
    messages = await node.prepare_messages([{'role': 'user', 'content': 'compare',
       'files': [attachment()], 'images': ['https://example.invalid/a.png']}])
    assert messages[0]['content'] == 'Attached file: notes.md\n\n# Notes\nHello'
    assert messages[1]['content'][0] == {'type': 'text', 'text': 'compare'}
    assert messages[1]['content'][1]['image_url']['url'] == 'https://example.invalid/a.png'
    node.inputs[node.INPUT_HANDLER_USER_IMAGES] = ['https://example.invalid/b.png']
    output = extract([item async for item in node.process(None)])
    assert output.messages[0]['content'][0]['image_url']['url'].endswith('/b.png')


@pytest.mark.asyncio
async def test_pdf_native_extraction_cached_by_hash_and_reused_across_history(monkeypatch):
    parser, paths = parser_mock(monkeypatch)
    cache = FakeRedis()
    node = NodeChat(ChatNodeModel(), file_cache=cache)
    document = attachment('report.pdf', b'%PDF-native-one', 'application/pdf')
    message = {'role': 'user', 'content': '', 'files': [document]}
    assert (await node.prepare_messages([message]))[0]['content'] == 'Attached file: report.pdf\n\nPDF text'
    await node.prepare_messages([message])
    assert parser.process_pdf.call_count == 1
    parser.process_pdf_with_ocr.assert_not_called()
    assert all(not Path(path).exists() for path in paths)
    key, value, ttl = cache.writes[0]
    assert key.endswith(hashlib.sha256(b'%PDF-native-one').hexdigest())
    assert value['markdown'] == 'PDF text'
    assert value['pdf_type'] == 'text_based'
    assert value['used_ocr'] is False
    assert ttl == 86400
    changed = attachment('report.pdf', b'%PDF-native-two', 'application/pdf')
    await node.prepare_messages([{**message, 'files': [changed]}])
    assert parser.process_pdf.call_count == 2
    assert len(cache.writes) == 2
    assert cache.writes[0][0] != cache.writes[1][0]


@pytest.mark.asyncio
async def test_pdf_uses_ocr_only_when_native_markdown_none_and_caches_ocr_metadata(monkeypatch):
    parser, paths = parser_mock(monkeypatch, markdown=None)
    cache = FakeRedis()
    node = NodeChat(ChatNodeModel(file_cache_ttl_seconds=37), file_cache=cache)
    result = await node.prepare_messages([{'role': 'user', 'content': 'scan',
        'files': [attachment('scan.pdf', b'%PDF-scanned', 'application/pdf')]}])
    assert result[0]['content'] == 'Attached file: scan.pdf\n\nOCR text'
    parser.process_pdf.assert_called_once()
    parser.process_pdf_with_ocr.assert_called_once_with(paths[0])
    assert cache.writes[0][1]['used_ocr'] is True
    assert cache.writes[0][1]['pages_routed_to_ocr'] == [1]
    assert cache.writes[0][2] == 37
    assert not Path(paths[0]).exists()


@pytest.mark.asyncio
async def test_missing_redis_warns_but_still_produces_chat(monkeypatch):
    parser_mock(monkeypatch)
    node = NodeChat(ChatNodeModel())
    node.inputs[node.INPUT_HANDLER_USER_FILES] = [attachment('native.pdf', b'%PDF-content', 'application/pdf')]
    outputs = [item async for item in node.process(None)]
    assert extract(outputs).messages[0]['content'].endswith('PDF text')
    warning = next(item['content'] for item in outputs if item['type'] == 'debug')
    assert warning['error_type'] == 'FileCacheWarning'
    assert warning['severity'] == 'warning'
    assert 'Redis' in warning['error_message']


@pytest.mark.asyncio
async def test_unavailable_redis_warns_and_corrupt_cache_is_replaced(monkeypatch):
    parser, _ = parser_mock(monkeypatch)
    cache = SimpleNamespace(get=AsyncMock(side_effect=ConnectionError('secret should not be logged')),
                            set=AsyncMock(side_effect=ConnectionError('secret should not be logged')))
    node = NodeChat(ChatNodeModel(), file_cache=cache)
    files = [attachment('native.pdf', b'%PDF-content', 'application/pdf')]
    await node.prepare_messages([{'role': 'user', 'content': '', 'files': files}])
    assert len(node._attachment_processor.warnings) == 2
    assert all('secret' not in item for item in node._attachment_processor.warnings)
    cache = FakeRedis()
    key = 'magic_agents:chat:pdf:v1:' + hashlib.sha256(b'%PDF-content').hexdigest()
    cache.values[key] = json.dumps({'version': 1, 'sha256': 'wrong', 'markdown': 'wrong cached text'})
    node = NodeChat(ChatNodeModel(), file_cache=cache)
    result = await node.prepare_messages([{'role': 'user', 'content': '', 'files': files}])
    assert result[0]['content'].endswith('PDF text')
    assert cache.writes[0][0] == key
    assert parser.process_pdf.call_count == 2


@pytest.mark.asyncio
async def test_redis_environment_configuration_closed_without_exposing_secrets(monkeypatch):
    parser_mock(monkeypatch)
    redis = FakeRedis()
    redis.aclose = AsyncMock()
    constructor = Mock(return_value=redis)
    monkeypatch.setitem(sys.modules, 'redis.asyncio', SimpleNamespace(Redis=SimpleNamespace(from_url=constructor)))
    monkeypatch.setenv('MAGIC_CHAT_TEST_REDIS_URL', 'redis://:private@localhost:6379/15')
    node = NodeChat(ChatNodeModel(file_cache_redis={'url_env': 'MAGIC_CHAT_TEST_REDIS_URL', 'ttl_seconds': 31}))
    await node.prepare_messages([{'role': 'user', 'content': '',
        'files': [attachment('native.pdf', b'%PDF-content', 'application/pdf')]}])
    constructor.assert_called_once_with('redis://:private@localhost:6379/15', socket_connect_timeout=2, socket_timeout=2)
    assert redis.writes[0][2] == 31
    redis.aclose.assert_awaited_once()
    assert 'private' not in str(node._capture_internal_state())
    await node.aclose()
    redis.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_missing_env_configuration_warns_and_parser_cleanup_on_ocr_failure(monkeypatch):
    parser, paths = parser_mock(monkeypatch, markdown=None)
    parser.process_pdf_with_ocr.side_effect = RuntimeError('runtime not installed')
    monkeypatch.delenv('MAGIC_CHAT_MISSING_REDIS', raising=False)
    node = NodeChat(ChatNodeModel(file_cache_redis={'url_env': 'MAGIC_CHAT_MISSING_REDIS'}))
    node.inputs[node.INPUT_HANDLER_USER_FILES] = [attachment('scan.pdf', b'%PDF-scanned', 'application/pdf')]
    outputs = [item async for item in node.process(None)]
    assert not any(item['type'] == node.OUTPUT_HANDLE for item in outputs)
    diagnostic = next(item for item in outputs if item.get('type') == 'debug')
    assert 'PDFium' in diagnostic['content']['error_message']
    assert not Path(paths[0]).exists()
    assert node._attachment_processor.warnings


@pytest.mark.asyncio
@pytest.mark.parametrize('descriptor,error', [
    (attachment('bad.md', b'\xff'), 'UTF-8'),
    (attachment('empty.txt', b' ', 'text/plain'), 'non-empty'),
    (attachment('null.txt', b'x\x00', 'text/plain'), 'UTF-8'),
    (attachment('pretend.pdf', b'not PDF', 'application/pdf'), 'PDF document'),
    ({**attachment(), 'data': 'data:text/markdown;base64,%%%%'}, 'data URI'),
    ({**attachment(), 'mime_type': 'application/pdf'}, 'MIME'),
    (attachment('unsupported.csv'), 'extensions'),
    (attachment('oversize.txt', b'x' * (1024 * 1024 + 1), 'text/plain'), 'size limit'),
    (attachment('large.pdf', b'%PDF-' + b'x' * (10 * 1024 * 1024), 'application/pdf'), 'size limit'),
])
async def test_malformed_and_oversized_file_descriptors_fail_closed(descriptor, error):
    node = NodeChat(ChatNodeModel())
    with pytest.raises(ChatFileError, match=error):
        await node.prepare_messages([{'role': 'user', 'content': 'hello', 'files': [descriptor]}])


@pytest.mark.asyncio
async def test_too_many_files_and_assistant_attachments_rejected():
    node = NodeChat(ChatNodeModel())
    with pytest.raises(ValueError, match='at most 4'):
        await node.prepare_messages([{'role': 'user', 'content': '', 'files': [attachment()] * 5}])
    with pytest.raises(ValueError, match='user message'):
        await node.prepare_messages([{'role': 'assistant', 'content': '', 'files': [attachment()]}])


@pytest.mark.asyncio
async def test_real_native_pdf_inspector_extracts_text_and_caches_without_ocr():
    pdf_inspector = pytest.importorskip('pdf_inspector')
    # Tiny valid PDF fixture, emitted without depending on another PDF library.
    stream = b'BT /F1 12 Tf 72 720 Td (Native chat attachment verification) Tj ET'
    objects = [b'<< /Type /Catalog /Pages 2 0 R >>', b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
        b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',
        b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
        b'<< /Length ' + str(len(stream)).encode() + b' >>\nstream\n' + stream + b'\nendstream']
    payload = b'%PDF-1.4\n'
    offsets = []
    for index, obj in enumerate(objects, 1):
        offsets.append(len(payload))
        payload += str(index).encode() + b' 0 obj\n' + obj + b'\nendobj\n'
    xref = len(payload)
    payload += b'xref\n0 6\n0000000000 65535 f \n'
    payload += b''.join(f'{offset:010d} 00000 n \n'.encode() for offset in offsets)
    payload += b'trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n' + str(xref).encode() + b'\n%%EOF\n'
    cache = FakeRedis()
    node = NodeChat(ChatNodeModel(), file_cache=cache)
    messages = await node.prepare_messages([{'role': 'user', 'content': '',
        'files': [attachment('native.pdf', payload, 'application/pdf')]}])
    assert 'Native chat attachment verification' in messages[0]['content']
    assert cache.writes[0][1]['used_ocr'] is False
    assert cache.writes[0][1]['pdf_type'] == 'text_based'
