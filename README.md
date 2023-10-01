# vLLM Client

## Overview

Client for the [vLLM](https://github.com/vllm-project/vllm) API with minimal dependencies.

## Examples

See [example.py](example.py) for the following:
- Single generation
- Streaming
- Batch inference

It should work out of the box with a vLLM API server. 

## Notes

- `sampling_params.py` needs to be kept in sync with vLLM.
  It is a simplified version of their class, containing
  only the code required on client side.
