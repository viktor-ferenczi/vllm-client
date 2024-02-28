import os
import unittest
import asyncio
import aiohttp

from vllm_client import AsyncVllmClient, SamplingParams

VLLM_BASE_URL = os.environ.get('VLLM_BASE_URL', 'http://127.0.0.1:8000')

# Suitable for Llama 2
PROMPT_TEMPLATE = '''\
<s>[INST] <<SYS>>
{system}
<</SYS>>

{instruction} [/INST]'''


class VllmClientTest(unittest.IsolatedAsyncioTestCase):
    """ Unit tests for the vLLM API client

    Requires a vLLM API server running on the api_base serving
    a model compatible with the prompt template below.

    """

    def setUp(self) -> None:
        super().setUp()

        self.client = AsyncVllmClient(VLLM_BASE_URL)

        self.prompt = PROMPT_TEMPLATE.format(
            system='Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n',
            instruction='You are a sci-fi fan who loves old fiction. What are your 10 favorite books?'
        )

    async def test_single_generation(self):
        n = 3
        count = 0
        params = SamplingParams(n=n, temperature=0.7, max_tokens=300)
        for completion in await self.client.generate(self.prompt, params):
            generated = completion[len(self.prompt):]
            print(generated)
            print()
            self.assertGreaterEqual(len(generated), 20)
            count += 1
        self.assertEqual(n, count)

    async def test_streaming_generation(self):
        received = self.prompt
        params = SamplingParams(temperature=0.7, max_tokens=300)
        async for completions in self.client.stream(self.prompt, params):
            output = completions[0]
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
        self.assertEqual(n, count)

    async def test_single_extra(self):
        n = 3
        count = 0
        params = SamplingParams(n=n, temperature=0.7, max_tokens=300)
        for completion in await self.client.generate(self.prompt, params, extra=dict(max_tokens=3)):
            generated = completion[len(self.prompt):]
            print(generated)
            print()
            count += 1
            self.assertLess(len(generated), 20)
        self.assertEqual(n, count)

    async def test_single_timeout(self):
        params = SamplingParams(n=16, temperature=0.7, max_tokens=3000)
        try:
            for _ in await self.client.generate(self.prompt, params, timeout=aiohttp.ClientTimeout(total=0.1)):
                self.fail('Did not time out')
        except asyncio.TimeoutError:
            self.assertTrue(True)
