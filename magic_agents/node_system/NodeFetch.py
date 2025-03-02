import json
import aiohttp
from jinja2 import Template
from magic_agents.node_system.Node import Node

class NodeFetch(Node):
    def __init__(self,
                 url: str,
                 method: str = 'POST',
                 data: dict = None,
                 headers: dict = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.method = method.upper()
        self.headers = headers or {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        self.url = url
        self.data = data or {}

    async def fetch(self, session, url, data):
        async with session.post(url, headers=self.headers, json=data) as response:
            response.raise_for_status()  # Lanzar excepción si falla la petición HTTP
            return await response.json()

    async def process(self, chat_log):
        # Renderizar dinámicamente el payload con Jinja
        template = Template(json.dumps(self.data))
        rendered_data = json.loads(template.render(self.inputs))

        async with aiohttp.ClientSession() as session:
            response_json = await self.fetch(session, self.url, rendered_data)

        yield {
            'type': 'end',
            'content': super().prep(response_json)
        }