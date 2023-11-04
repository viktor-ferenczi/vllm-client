from json import loads
from typing import AsyncIterable, List

import aiohttp

from vllm_client.sampling_params import SamplingParams


class AsyncVllmClient:

    def __init__(self, url: str):
        url = url.rstrip('/')
        if url.endswith('/generate'):
            raise ValueError('Please remove /generate from the end of API URL')

        self.url: str = url
        self.__generate_url = f'{url}/generate'

    async def generate(self, prompt: str, params: SamplingParams) -> List[str]:
        payload = {
            "prompt": prompt,
            **params.__dict__
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.__generate_url, json=payload) as response:
                response.raise_for_status()
                response = await response.json()

        return response["text"]

    async def stream(self, prompt: str, params: SamplingParams) -> AsyncIterable[List[str]]:
        payload = {
            "prompt": prompt,
            "stream": True,
            **params.__dict__
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.__generate_url, json=payload) as response:
                content = response.content
                while 1:
                    item = await content.readuntil(b"\0")
                    if not item:
                        break
                    yield loads(item[:-1].decode("utf-8"))["text"]
