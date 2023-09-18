# vLLM Client

## Overview

Client for the vLLM API with minimal dependencies.

## Examples

See [example.py](example.py) for the following:
- Single generation
- Streaming
- Batch inference

It should work out of the box with a vLLM API server 
running a Llama-2 model (any parameter count). 

## Notes

- `sampling_params.py` is a copy of the file with the same name
  from the vLLM repository. It needs to be kept in sync.
