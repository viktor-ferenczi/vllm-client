#!/usr/bin/python3
""" Example for using the vLLM API Client
"""
import asyncio

from vllm_client.async_client import AsyncVllmClient
from vllm_client.sampling_params import SamplingParams

# Adjust to match your vLLM server
API_BASE = 'http://127.0.0.1:8000/generate'

# Adjust to match the model running on the vLLM server
# This one works for Llama-2
PROMPT_TEMPLATE = '''\
<s>[INST] <<SYS>>
{system}
<</SYS>>

{instruction} [/INST]'''

SYSTEM = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.'


async def main():
    print('=== Initialization ===')

    client = AsyncVllmClient(API_BASE)

    # See the documentation in the docstring of the SamplingParams class
    params = SamplingParams(n=3, temperature=0.7, max_tokens=200)

    prompt = PROMPT_TEMPLATE.format(
        system=SYSTEM,
        instruction='You are a sci-fi fan who loves old fiction. What are your favorite books?'
    )

    print('=== Single generation ===')
    print()

    print(prompt)
    print()

    response = await client.generate(prompt, params)

    for i, output in enumerate(response):
        print(f'Output #{1 + i}:')
        print(output[len(prompt):])
        print()

    print('=== Streaming generation ===')
    print()

    print(prompt)
    print()

    received = prompt
    async for d in client.stream(prompt, params):
        # NOTE: Print only the first output stream for clarity
        output = d[0]
        print(output[len(received):], end='')
        received = output
    print()
    print()

    print('=== Parallel generation ===')
    print()

    instructions = [
        'You are a PC enthusiast. What components your PC contains and why?',
        'You design rockets. What are the most important design principles?',
        'You own a flower shop. What kind of flowers do you sell and why?',
    ]

    prompts = [
        PROMPT_TEMPLATE.format(system=SYSTEM, instruction=i)
        for i in instructions
    ]

    tasks = [client.generate(p, params) for p in prompts]
    responses = await asyncio.gather(*tasks)

    for i, response in enumerate(responses):
        print(f'Prompt #{1 + i}:')
        print(prompts[i])
        print()
        for j, output in enumerate(response):
            print(f'Prompt #{1 + i} / Output #{1 + j}:')
            print(output[len(prompts[i]):])
            print()


if __name__ == '__main__':
    asyncio.run(main())
