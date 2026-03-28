# radixInfer

`radixInfer` is a refactored inference service skeleton that keeps the major request lifecycle of `mini-sglang` while replacing the code structure with clearer layers:

- `api`: HTTP and SSE streaming
- `transport`: queues, protocol, tokenizer worker
- `runtime`: request state machine, planner, scheduler
- `cache`: logical page allocator and prefix cache
- `engine`: pluggable generation backends

This initial implementation focuses on the new architecture and a runnable end-to-end control plane.
