import time
import json

from magic_agents.node_system.Node import Node


class NodeSendMessage(Node):
    def __init__(self,
                 text: str = '',
                 extras: dict = None
                 ) -> None:
        super().__init__()
        self._text = text
        self._extras = extras

    async def __call__(self, chat_log) -> dict:
        print('Node text')
        lt = {
            'id': -1,
            'id_chat': chat_log.id_chat,
            'choices':
                [{
                    'delta':
                        {
                            'content': self._text,
                            'role': None
                        },
                    'finish_reason': None,
                    'index': 0
                }],
            'created': int(time.time()),
            'model': '',
            'usage': {
                'prompt_tokens': thread_log.thread_content_token_context,
                'completion_tokens': thread_log.thread_content_token_generated
            },
            'object': 'chat.completion.chunk',
            'extras': self._extras
        }

        yield 'data: ' + json.dumps(lt, separators=(',', ':')) + '\n\n'
