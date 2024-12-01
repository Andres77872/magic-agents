import json
from concurrent.futures import ThreadPoolExecutor

import aiohttp
from jinja2 import Template

from magic_agents.node_system.Node import Node


async def make_request(url, data, headers):
    if headers is None:
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/x-www-form-urlencoded'
        }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=data) as response:
            json_response = await response.json()
            return json_response


def sync_make_request(url: str, data: dict, headers: dict = None):
    import asyncio
    return asyncio.run(make_request(url, data, headers))


class NodeFetch(Node):
    def __init__(self,
                 url: str,
                 method: str = 'POST',
                 data: dict = None,
                 headers: dict = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.method = method
        self.headers = headers
        self.url = url
        self.data = data

    async def __call__(self, chat_log) -> dict:
        data_str = json.dumps(self.data)
        template = Template(data_str)
        output = template.render(self.parents)
        data = json.loads(output)

        with ThreadPoolExecutor() as executor:
            res = executor.submit(sync_make_request, self.url, data, self.headers).result()

        return super().prep(res)
