from json import loads, dumps
from logging import Logger
from typing import AsyncIterable, List, Optional, Dict, Any

import aiohttp

from .sampling_params import SamplingParams


class AsyncVllmClient:

    def __init__(self, url: str, *, logger: Optional[Logger] = None):
        url = url.rstrip('/')
        if url.endswith('/generate'):
            raise ValueError('Please remove /generate from the end of API URL')

        self.logger: Optional[Logger] = logger

        self.url: str = url
        self.__generate_url = f'{url}/generate'

    async def generate(self,
                       prompt: str,
                       params: SamplingParams,
                       extra: Optional[Dict[str, Any]] = None) -> List[str]:
        payload = {
            "prompt": prompt,
            **params.__dict__
        }

        if extra is not None:
            payload.update(extra)

        if self.logger is not None:
            self.logger.debug('vLLM request')
            self.logger.debug(f'url: {self.__generate_url}')
            self.logger.debug(f'payload:\n{dumps(payload, indent=2)}')

        async with aiohttp.ClientSession() as session:
            async with session.post(self.__generate_url, json=payload) as response:
                response.raise_for_status()
                response = await response.json()

        if self.logger is not None:
            self.logger.debug(f'response:\n{dumps(response, indent=2)}')

        return response["text"]

    async def stream(self,
                     prompt: str,
                     params: SamplingParams,
                     extra: Optional[Dict[str, Any]] = None) -> AsyncIterable[List[str]]:
        payload = {
            "prompt": prompt,
            "stream": True,
            **params.__dict__
        }

        if extra is not None:
            payload.update(extra)

        async with aiohttp.ClientSession() as session:
            async with session.post(self.__generate_url, json=payload) as response:
                content = response.content
                while 1:
                    item = await content.readuntil(b"\0")
                    if not item:
                        break
                    yield loads(item[:-1].decode("utf-8"))["text"]
