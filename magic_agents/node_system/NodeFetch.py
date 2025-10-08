import json
import logging

import aiohttp
from jinja2 import Template
from urllib.parse import urlsplit

from magic_agents.models.factory.Nodes import FetchNodeModel
from magic_agents.node_system.Node import Node

logger = logging.getLogger(__name__)


class NodeFetch(Node):
    def __init__(self,
                 data: FetchNodeModel,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.method = data.method.upper().strip()
        self.headers = data.headers or {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        self.url = data.url
        self.data = data.data or None
        # Add jsondata attribute if it exists in the model
        self.jsondata = getattr(data, 'json_data', None)
        if not self.jsondata:
            self.jsondata = None

    async def fetch(self, session, url, data=None, json_data=None):
        # Use the appropriate method (GET, POST, PUT, etc.)
        method = session.request
        kwargs = {
            'method': self.method,
            'url': url,
            'headers': self.headers if type(self.headers) is dict else json.loads(self.headers)
        }

        # Add data based on what's available
        if json_data is not None:
            json_data = json_data if type(json_data) is dict else json.loads(json_data)
            kwargs['json'] = json_data
        elif data is not None:
            data = data if type(data) is dict else json.loads(data)
            kwargs['data'] = data

        if 'json' not in kwargs and 'data' not in kwargs:
            if self.method != 'GET':
                return {}

        parts = urlsplit(url)
        safe_url = f"{parts.scheme}://{parts.netloc}{parts.path}"

        logger.info("NodeFetch:%s %s %s", self.node_id, self.method, safe_url)
        if self.debug:
            payload_type = 'json' if 'json' in kwargs else ('data' if 'data' in kwargs else 'none')
            logger.debug("NodeFetch:%s request payload type=%s headers_keys=%s", self.node_id, payload_type, list(kwargs['headers'].keys()))

        async with method(**kwargs) as response:
            if self.debug:
                logger.debug("NodeFetch:%s response status=%s", self.node_id, response.status)
            response.raise_for_status()
            return await response.json()

    async def process(self, chat_log):
        # Prepare the data to send
        data_to_send = None
        json_data_to_send = None
        run = False
        for i in self.inputs.values():
            if i:
                run = True
                break
        if not run:
            if self.debug:
                logger.debug("NodeFetch:%s no inputs set; skipping request", self.node_id)
            yield self.yield_static({})
            return
        if self.jsondata is not None:
            # Render JSON data with Jinja if jsondata exists
            template = Template(json.dumps(self.jsondata))
            json_data_to_send = json.loads(template.render(self.inputs))
        elif self.data:
            # Render regular data with Jinja if data exists
            template = Template(json.dumps(self.data))
            data_to_send = json.loads(template.render(self.inputs).replace('\n', ''))

        async with aiohttp.ClientSession() as session:
            logger.debug("NodeFetch:%s executing fetch", self.node_id)
            response_json = await self.fetch(
                session,
                self.url,
                data=data_to_send,
                json_data=json_data_to_send
            )
        logger.info("NodeFetch:%s request completed", self.node_id)
        yield self.yield_static(response_json)
