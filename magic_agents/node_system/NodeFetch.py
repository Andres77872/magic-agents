import json

import aiohttp
from jinja2 import Template

from magic_agents.models.factory.Nodes import FetchNodeModel
from magic_agents.node_system.Node import Node


class NodeFetch(Node):
    def __init__(self,
                 data: FetchNodeModel,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.method = data.method.upper()
        self.headers = data.headers or {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        self.url = data.url
        self.data = data.data or None
        # Add jsondata attribute if it exists in the model
        self.jsondata = getattr(data, 'json_data', None)

    async def fetch(self, session, url, data=None, json_data=None):
        # Use the appropriate method (GET, POST, PUT, etc.)
        method = session.request
        kwargs = {
            'method': self.method,
            'url': url,
            'headers': self.headers
        }

        # Add data based on what's available
        if json_data is not None:
            kwargs['json'] = json_data
        elif data is not None:
            kwargs['data'] = data
        async with method(**kwargs) as response:
            response.raise_for_status()
            return await response.json()

    async def process(self, chat_log):
        # Prepare the data to send
        data_to_send = None
        json_data_to_send = None

        if self.jsondata is not None:
            # Render JSON data with Jinja if jsondata exists
            template = Template(json.dumps(self.jsondata))
            json_data_to_send = json.loads(template.render(self.inputs))
        elif self.data:
            # Render regular data with Jinja if data exists
            template = Template(json.dumps(self.data))
            data_to_send = json.loads(template.render(self.inputs))

        async with aiohttp.ClientSession() as session:
            response_json = await self.fetch(
                session,
                self.url,
                data=data_to_send,
                json_data=json_data_to_send
            )
        yield self.yield_static(response_json)
