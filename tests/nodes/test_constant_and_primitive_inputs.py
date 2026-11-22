import pytest
from unittest.mock import MagicMock, patch

from magic_agents.models.factory.Nodes import (
    ClientNodeModel,
    ConstantNodeModel,
    FetchNodeModel,
    LlmNodeModel,
    PythonExecNodeModel,
)
from magic_agents.models.model_agent_run_log import ModelAgentRunLog
from magic_agents.node_system import NodeClientLLM, NodeConstant, NodeFetch, NodeLLM, NodePythonExec


class TestConstantNodeModel:
    def test_parses_int_value(self):
        model = ConstantNodeModel(value_type='int', value='42')
        assert model.value == 42

    def test_parses_bool_value(self):
        model = ConstantNodeModel(value_type='bool', value='true')
        assert model.value is True

    def test_defaults_none_for_string(self):
        model = ConstantNodeModel(value_type='str', value=None)
        assert model.value == ''

    def test_rejects_invalid_float(self):
        with pytest.raises(ValueError):
            ConstantNodeModel(value_type='float', value='nope')


class TestNodeConstant:
    @staticmethod
    async def collect_outputs(node):
        return [
            item
            async for item in node.process(ModelAgentRunLog())
            if item['type'] != 'content'
        ]

    @pytest.mark.asyncio
    async def test_process_yields_constant_output(self):
        node = NodeConstant(
            data=ConstantNodeModel(value_type='int', value='5'),
            node_id='const-1',
            debug=False,
        )

        outputs = await self.collect_outputs(node)

        assert outputs == [
            {
                'type': 'handle_constant_output',
                'content': {'node': 'NodeConstant', 'content': 5},
            }
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'value_type,raw_value,expected_value,expected_python_type',
        [
            ('bool', 'true', True, bool),
            ('str', 42, '42', str),
            ('float', '0.85', 0.85, float),
        ],
    )
    async def test_process_emits_each_declared_primitive_type(
        self,
        value_type,
        raw_value,
        expected_value,
        expected_python_type,
    ):
        node = NodeConstant(
            data=ConstantNodeModel(value_type=value_type, value=raw_value),
            node_id=f'const-{value_type}',
            debug=False,
        )

        outputs = await self.collect_outputs(node)

        assert outputs == [
            {
                'type': 'handle_constant_output',
                'content': {
                    'node': 'NodeConstant',
                    'content': expected_value,
                },
            }
        ]
        assert type(outputs[0]['content']['content']) is expected_python_type

    @pytest.mark.asyncio
    @pytest.mark.parametrize('handle_key', ['output', 'value'])
    async def test_process_emits_on_custom_output_handle_aliases(self, handle_key):
        node = NodeConstant(
            data=ConstantNodeModel(value_type='int', value='7'),
            handles={handle_key: 'custom-constant'},
            node_id=f'const-{handle_key}',
            debug=False,
        )

        outputs = await self.collect_outputs(node)

        assert outputs == [
            {
                'type': 'custom-constant',
                'content': {'node': 'NodeConstant', 'content': 7},
            }
        ]

    def test_capture_internal_state_includes_typed_value(self):
        node = NodeConstant(
            data=ConstantNodeModel(value_type='float', value='1.25'),
            node_id='const-debug-state',
            debug=False,
        )

        state = node._capture_internal_state()

        assert state['value'] == 1.25
        assert state['value_type'] == 'float'


class TestPrimitiveInputConsumption:
    def test_llm_uses_connected_primitive_inputs(self):
        node = NodeLLM(
            data=LlmNodeModel(stream=False, json_output=False, iterate=False, temperature=0.1, top_p=0.2, max_tokens=100),
            node_id='llm-1',
            debug=False,
        )
        node.inputs.update({
            'handle-llm-stream': 'true',
            'handle-llm-json_output': '1',
            'handle-llm-iterate': 'yes',
            'handle-llm-temperature': '0.9',
            'handle-llm-top_p': '0.75',
            'handle-llm-max_tokens': '256',
        })

        node.stream = node._resolve_runtime_value(node.INPUT_HANDLER_STREAM, node._default_stream, 'bool')
        node.json_output = node._resolve_runtime_value(node.INPUT_HANDLER_JSON_OUTPUT, node._default_json_output, 'bool')
        node.iterate = node._resolve_runtime_value(node.INPUT_HANDLER_ITERATE, node._default_iterate, 'bool')
        node.extra_data = node._build_runtime_extra_data()

        assert node.stream is True
        assert node.json_output is True
        assert node.iterate is True
        assert node.extra_data['temperature'] == 0.9
        assert node.extra_data['top_p'] == 0.75
        assert node.extra_data['max_tokens'] == 256

    def test_fetch_uses_connected_request_overrides(self):
        node = NodeFetch(
            data=FetchNodeModel(url='https://default.example.com', method='GET', headers={'X-Base': '1'}, data='base', json_data={'base': True}),
            node_id='fetch-1',
            debug=False,
        )
        node.inputs.update({
            'handle-url': 'https://override.example.com',
            'handle-fetch-method': 'post',
            'handle-fetch-headers': {'Authorization': 'Bearer test'},
            'handle-fetch-data': '{"hello": "world"}',
            'handle-fetch-json_data': {'ok': True},
        })

        url, method, headers, data, jsondata = node._resolve_runtime_request_config()

        assert url == 'https://override.example.com'
        assert method == 'POST'
        assert headers == {'Authorization': 'Bearer test'}
        assert data == '{"hello": "world"}'
        assert jsondata == {'ok': True}

    @pytest.mark.asyncio
    async def test_client_reinitializes_with_connected_engine_and_model(self):
        clients = [MagicMock(name='authored'), MagicMock(name='routed')]
        with patch('magic_agents.node_system.NodeClientLLM.MagicLLM', side_effect=clients) as constructor:
            node = NodeClientLLM(data=ClientNodeModel(engine='openai', model='approved'), node_id='client-1')
            node.inputs.update({'handle-client-engine': 'anthropic', 'handle-client-model': 'another'})

            events = [event async for event in node.process(None)]

        assert [call.kwargs['engine'] for call in constructor.call_args_list] == ['openai', 'anthropic']
        assert [call.kwargs['model'] for call in constructor.call_args_list] == ['approved', 'another']
        assert [e['content']['content'] for e in events if e['type'] == node.OUTPUT_HANDLE] == [clients[1]]

    def test_python_exec_uses_connected_runtime_config(self):
        node = NodePythonExec(
            data=PythonExecNodeModel(safety_mode='subprocess', timeout=30.0, max_output_chars=8000),
            node_id='py-1',
            debug=False,
        )
        node.inputs.update({
            'handle-python_exec-safety_mode': 'restricted_builtins',
            'handle-python_exec-timeout': '12.5',
            'handle-python_exec-max_output_chars': '2048',
        })

        safety_mode, timeout, max_output_chars = node._resolve_runtime_config()
        node._refresh_executor(safety_mode, timeout, max_output_chars)

        assert node.executor.safety_mode == 'restricted_builtins'
        assert node.executor.timeout == 12.5
        assert node.executor.max_output_chars == 2048
