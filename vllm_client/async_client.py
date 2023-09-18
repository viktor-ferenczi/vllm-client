from json import loads
from typing import AsyncIterable, List

import aiohttp

from .sampling_params import SamplingParams


class AsyncVllmClient:

    def __init__(self, url: str):
        self.url: str = url

    async def generate(self, prompt: str, params: SamplingParams) -> List[str]:
        payload = {
            "prompt": prompt,
            **params.__dict__
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, json=payload) as response:
                response.raise_for_status()
                response = await response.json()

        return response['text']

    async def stream(self, prompt: str, params: SamplingParams) -> AsyncIterable[List[str]]:
        payload = {
            "prompt": prompt,
            "stream": True,
            **params.__dict__
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, json=payload) as response:
                content = response.content
                while 1:
                    item = await content.readuntil(b'\0')
                    if not item:
                        break
                    yield loads(item[:-1].decode('utf-8'))['text']
