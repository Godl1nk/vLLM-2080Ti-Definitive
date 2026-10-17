# CPU overhead on GPU serving hosts

English | [简体中文](cpu-tuning.zh-CN.md)

This fork is based on upstream vLLM and vLLM 2080 Ti Definitive Edition by
`github.com/weicj`. GPU inference still runs HTTP handling, tokenization,
scheduling, metadata preparation, kernel dispatch and streaming on the CPU.
Extra CPU cores let separate processes run together; they do not make one
Python scheduling/dispatch loop execute on all cores simultaneously.

## Runtime changes

Compiled FULL graph metadata refresh skips consecutive references to the same
source and destination metadata. Layers in an attention group share metadata,
so repeating the traversal and copies for those layers is unnecessary. Only
consecutive aliases are skipped: intervening writes preserve their original
order. Both graph paths skip tensor self-copies. V2 refresh also skips topology
and container checks on built-in scalar leaves; parent topology validation and
tensor shape/dtype/device checks remain active.

Batch planning also skips reordering for zero/one request and counts the four
request regions in one pass instead of four separate reductions. Small batches
of up to 16 requests use one CPU offset read and Python integer comparisons to find
the decode/prefill boundary. Larger batches and explicit prefill masks retain
the vectorized path. Padded token counts and uniform-decode rules are preserved.
The reorder shortcut applies to runners using the shared reorder utility;
it does not add that utility to runners with their own ordering logic.

These are CPU staging changes, not new graph modes or kernel policies. CPU
tensor regression tests cover buffer identity, shared and conflicting aliases,
and topology/tensor/container errors. End-to-end RTX 2080 Ti validation is still
required before claiming a serving speedup or promoting these changes as a
validated deployment release.

## Xeon E5 v4 host setup

- Leave `OMP_NUM_THREADS` unset unless you intentionally override vLLM. This
  checkout already uses CPU parallelism during startup and reduces Torch
  intra-op threads to one for steady-state GPU serving. Forcing 40 or 80
  threads into tiny CPU operations can create contention. A/B test explicit
  `OMP_NUM_THREADS=1` if another service configuration forces a larger value;
  this also limits startup parallelism.
- Keep at least four physical cores available for TP2: API, engine and two GPU
  workers, plus room for OS/background work. Do not pin all processes onto one
  core or two sibling hyperthreads. More API workers help simultaneous input
  processing, not the serial decode loop of one chat.
- On multi-socket hosts, inspect `nvidia-smi topo -m`, `numactl --hardware` and
  `lscpu -e=CPU,NODE,SOCKET,CORE`. A/B test adding `--numa-bind` to an existing
  direct `vllm serve` command, with `numactl` installed. vLLM auto-detects GPU
  locality. Do not guess CPU/node numbers; manual `--numa-bind-cpus` and
  `--numa-bind-nodes` require the actual topology. Keep API/background tasks
  from starving workers. NUMA flags are vLLM CLI options, not route-profile
  fields. See [upstream NUMA guidance](https://docs.vllm.ai/en/latest/configuration/optimization/#numa-binding-for-multi-socket-gpu-nodes).
- A/B test `--stream-interval 4` if streaming/HTTP CPU work is material. It batches
  token delivery and can reduce host overhead, at the cost of less granular
  streaming. Measure client-visible latency as well as throughput; leave it at
  one when immediate per-token delivery matters.
- Keep the route's validated graph and speculative-decoding settings. Do not
  force eager execution or enable unsafe synchronization modes to reduce CPU
  utilization. Avoid request/output logging and tracing during timed runs.

These settings are experiments, not an E5-2698 v4 performance guarantee. A
CPU bottleneck cannot be established from low GPU utilization alone.

## Repeatable checks

Run from the checkout using a Python environment with Torch:

```bash
python -B tests/standalone_tests/test_metadata_cpu.py
python -B tests/standalone_tests/test_attention_cpu.py
python -B tools/benchmark_metadata_cpu.py --baseline-ref 93c9013e37
python -B tools/benchmark_metadata_cpu.py --suite attention --baseline-ref 93c9013e37
```

The benchmark loads the actual source helper definitions without importing
vLLM's GPU extensions. It uses real CPU tensors, one Torch thread, 48 synthetic
layer aliases, warmup and median timing over seven repeats of 1,000 calls.
It reports both a shared-layer case and an isolated-metadata case so traversal
overhead is visible. This is not a model, Xeon or CUDA benchmark; it excludes
GPU transfer, kernel execution, scheduling and HTTP costs. Use another Git
baseline explicitly if comparing later changes.

The attention suite times single and multi-request reordering, speculative
splitting, plain-decode early returns and mixed prefill/decode batches. Its
unsorted-reorder cases include identical fixture-reset costs on both versions
because reordering mutates the batch. The CPU-only scope limitation applies
to this suite as well.

On the target host, compare the same checkpoint, profile, mode, KV precision,
speculative settings, prompts, token counts and GPU clocks before/after. Follow
the reference lane in `profiles/2x2080Ti/README.md`: exclude warmup, disable
prefix caching for the reference measurement, use three 4K/128 requests and a
completed 32K/512 request, and check answers and exact token counts. Keep raw
results outside tracked source. Use [Nsight Systems](https://docs.vllm.ai/en/latest/contributing/profiling/)
to distinguish CPU dispatch gaps from GPU compute/communication time. Do not
treat tracing overhead as production latency.
