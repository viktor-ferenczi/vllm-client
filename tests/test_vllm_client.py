import unittest
import asyncio

from vllm_client.sampling_params import SamplingParams
from vllm_client.async_client import AsyncVllmClient


class VllmClientTest(unittest.IsolatedAsyncioTestCase):
    """ Unit tests for the vLLM API client

    Requires a vLLM API server running on the api_base serving
    a model compatible with the prompt template below.

    """

    api_base = 'http://127.0.0.1:8000/generate'

    prompt_template = '''\
<s>[INST] <<SYS>>
{system}
<</SYS>>

{instruction} [/INST]'''

    def setUp(self) -> None:
        super().setUp()

        self.client = AsyncVllmClient(self.api_base)

        self.prompt = self.prompt_template.format(
            system='Below is an instruction that describes a task. Write a response that appropriately completes the request.',
            instruction='You are a sci-fi fan who loves old fiction. What are your favorite books?'
        )

    async def test_single_generation(self):
        n = 3
        count = 0
        for s in await self.client.generate(self.prompt, SamplingParams(n=n, temperature=0.7, max_tokens=300)):
            print(s[len(self.prompt):])
            print()
            count += 1
        self.assertEquals(n, count)

    async def test_streaming_generation(self):
        received = self.prompt
        async for d in self.client.stream(self.prompt, SamplingParams(temperature=0.7, max_tokens=300)):
            output = d[0]
            print(output[len(received):], end='')
            received = output
        self.assertGreater(len(received), len(self.prompt) + 100)

    async def test_parallel_generation(self):
        n = 10
        params = SamplingParams(temperature=0.5, max_tokens=300)
        tasks = [self.client.generate(self.prompt, params) for _ in range(n)]
        responses = await asyncio.gather(*tasks)
        count = 0
        for response in responses:
            print(response[0][len(self.prompt):])
            print()
            count += 1
        self.assertEquals(n, count)
