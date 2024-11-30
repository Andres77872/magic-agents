import json
from concurrent.futures import ThreadPoolExecutor

import aiohttp
from jinja2 import Template

from magic_agents.node_system.Node import Node
from magic_agents.util.const import HANDLE_FETCH_JSON_INPUT, HANDLE_FETCH_TEXT_INPUT


async def make_request(url, data, headers):
    if headers is None:
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/x-www-form-urlencoded'
        }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=data) as response:
            print(f"Status: {response.status}")
            json_response = await response.json()
            print(json_response)
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
        print('Node Fetch', self.parents)
        print('Node Fetch', self.method)
        print('Node Fetch', self.url)
        print('Node Fetch', self.data)

        data_str = json.dumps(self.data)
        print('Node Fetch:data', type(data_str), data_str)
        template = Template(data_str)
        output = template.render(self.parents)
        data = json.loads(output)
        print('Node Fetch:data', data)

        with ThreadPoolExecutor() as executor:
            res = executor.submit(sync_make_request, self.url, data, self.headers).result()

        print('Node Fetch Response: ', res)
        return super().prep(res)
