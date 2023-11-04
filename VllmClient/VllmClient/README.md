# vLLM Client

## Overview

Client for the [vLLM](https://github.com/vllm-project/vllm) API with minimal dependencies.

## Installation

Install `vLLM Client` from NuGet.

## Examples

```csharp
const string ApiUrl = "http://localhost:8000";
var client = new AsyncVllmClient(ApiUrl);

// See the docstring of the SamplingParams [*] Python class 
var @params = new SamplingParams { Temperature = 0.5f, MaxTokens = 2 };

const string prompt = "Hello! My name is";

// Full generation (list with N items, see SamplingParams)
IList<string> response = await client.Generate(prompt, @params);

// Streaming generation
await foreach (var response in client.Stream(prompt, @params)) {
    // Response is an IList of N strings
}
```

`*` [SamplingParams](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py)

See also the unit tests in the repository as examples.

Make sure to consider the prompt template of the model used (if any) to avoid confusion.

## Notes

- `SamplingParams` needs to be kept in sync with vLLM.
