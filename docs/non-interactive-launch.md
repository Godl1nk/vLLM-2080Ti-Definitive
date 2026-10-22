# Non-Interactive Launch

Build the runtime with `./build.sh`, then pass a flat profile path. Profile
files contain route parameters only; checkpoint, GPU topology, port, and mode
remain launcher options.

```bash
./launcher.sh \
  --model-dir /mnt/models/Qwen3.8-27B-FP8 \
  --profile 2x2080Ti/qwen27b/w8a16/mtp4-fp16kv-1x148K-text-only.env \
  --mode fast \
  --gpu-devices 1,5 \
  --tp-size 2 \
  --pp-size 1 \
  --print-config
```

`--mode` is optional and defaults to `fast`. A profile without `MODE` keeps
the launcher selection; an explicit profile `MODE=normal` or `MODE=fast` may
override it. There are no `fast/` or `normal/` profile directories.

Useful options include `--model-dir`, `--speculative-model`, `--profile`,
`--mode`, `--gpu-devices`, `--tp-size`, `--pp-size`, `--port`,
`--start-timeout`, and `--print-config`. Use `--set KEY=VALUE` for advanced
launcher/runtime settings. Do not add Prefix Cache, Mamba cache, GPU, port, or
model-path fields to a profile; the validator rejects them.

The profile library is a validation matrix, not a promise that every filename
fits every machine. Capacity and performance are promoted only after the
external audit records startup, 4K/128, 32K/512, concurrency, and image
correctness evidence.

## vLLM CLI Style Serving

`vllm-server` is a runnable entry point that uses vLLM CLI flag names, mirrors
the `vllm serve` positional model tag, and serves in the foreground; press
Ctrl+C to stop. The model is an HF-format checkpoint directory
(`config.json` + `*.safetensors`). Any launcher option (for example `--profile`,
`--mode`, or `--gpu-devices`) passes through unchanged.

When no route profile or route flags are given, the launcher auto-selects the
validated profile for the detected GPUs, model family, and weight quantization.
For example, an FP8 Qwen3.8 checkpoint on two RTX 2080 Ti resolves to
`2x2080Ti/qwen27b/w8a16/mtp4-fp8kv-1x186K-text-image.env` (186K context, FP8
KV, MTP4, TP2). Explicit flags and `--profile` take precedence. Auto-selection
requires `nvidia-smi`; unsupported hardware falls back to the built-in
defaults.

```bash
./vllm-server \
  --served-model-name qwen3.8-27b \
  --max-model-len 32768 \
  --kv-cache-dtype float16 \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 \
  --port 8080 \
  /mnt/models/Qwen3.8-27B-FP8
```

Supported vLLM options and mappings:

- The positional model tag and `--model` both select the checkpoint
  (`MODEL_DIR`, `vllm serve` style; only one model is accepted).
- `--served-model-name` sets `SERVED_NAME`.
- `--max-model-len`, `--max-num-seqs`, and `--max-num-batched-tokens` map to
  `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, and `MAX_BATCHED_TOKENS`.
- `--gpu-memory-utilization` maps to `GPU_UTIL` (a fraction between 0 and 1).
- `--kv-cache-dtype` validates against the fork's KV precisions:
  `float16`, `int8_per_token_head`, `fp8`, `turboquant_k8v4`,
  `turboquant_4bit_nc`; `auto` selects `float16`.
- `--tensor-parallel-size` and `--pipeline-parallel-size` map to `TP_SIZE` and
  `PP_SIZE`; `--chat-template` maps to `CHAT_TEMPLATE_FILE`.
- `--host` maps to `SERVICE_SCOPE`: loopback addresses use local scope,
  everything else binds `0.0.0.0` on the LAN scope.
- `./launcher.sh` accepts the same flags directly; it starts the daemonized
  service-manager flow instead of a foreground server.

Unsupported options fail fast with the validated alternative in the message.
GGUF/ggml checkpoints are rejected; this runtime loads HF-format checkpoints
and its own `--quantization` schemes.

## Experimental SSD Prefix Cache

The launcher can opt into vLLM's built-in `OffloadingConnector` with a CPU
staging tier and a filesystem tier. This stores completed prefix KV blocks on
disk without a separate cache daemon. Keep the directory on persistent storage
and use the same model path, KV precision, block layout, and parallel settings
after restarting. The CPU staging allocation uses host RAM; it is not GPU KV
capacity. On each launch, the launcher hashes every file in the target and
local draft checkpoints before loading the model. The content digest selects
a separate disk namespace, so replacing weights at the same path cannot load
old KV. This adds one full checkpoint read to startup; old namespaces must be
removed manually to reclaim space. Clear the cache after changing to an
incompatible runtime. Disk caching requires local checkpoint directories;
do not modify checkpoint files while the launcher starts or the service runs.

The filesystem tier does not evict old files or enforce a disk quota. Use a
dedicated volume with sufficient free space, monitor its usage, and clear stale
cache data while the service is stopped. `KV_DISK_CPU_BYTES` limits only host
RAM staging. The launcher fixes `PYTHONHASHSEED=0` when disk caching is enabled
and rejects another value, so block hashes stay stable across restarts.

```bash
./launcher.sh \
  --model-dir /mnt/models/Qwen3.8-27B-AWQ-INT4 \
  --gpu-devices 0,2,3,4 --tp-size 4 \
  --set KV_DISK_CACHE_DIR=/mnt/nvme/vllm-kv-cache \
  --set KV_DISK_CPU_BYTES=4294967296 \
  --print-config
```

Remove `--print-config` to start the service. This option requires prefix
caching and sets a stable `PYTHONHASHSEED`. Before stopping the service, wait
for `vllm:kv_offload_tiering_active_cascade_jobs` to return to zero. Compare
`vllm:external_prefix_cache_hits_total` and the filesystem tier's
`vllm:kv_offload_tiering_chunk_hits_total` and
`vllm:kv_offload_tiering_read_bytes_total` after restarting.

On 4xT10 with a Qwen3.8-27B AWQ-INT4 checkpoint and FP16 KV, an identical
1,929-token prompt recovered 1,568 tokens from SSD after a full host reboot;
the answer matched the cold run. The FS tier reported five hits and 257 MB
read. This route remains experimental and has no published throughput claim.
On dual RTX 2080 Ti, the same model and FP16 KV recovered 1,568 of 1,933
prompt tokens after the vLLM service was stopped and restarted, with the same
answer. The dual-2080-Ti run did not include a host reboot.

The shipped `fast` profiles were also checked on dual RTX 2080 Ti (GPUs 1,5)
with the Qwen3.8-27B checkpoints. The W8A16 FP8-weight / FP8-KV MTP4 262K
text-only profile recovered 3,232 of 5,356 prompt tokens after a service
restart with 4 GiB of CPU staging. The W4A16 NVFP4-weight / FP8-KV DFlash2
262K text-and-image profile recovered 3,296 of 5,355 text prompt tokens after
a service restart with 8 GiB of CPU staging; the filesystem tier reported 21
chunk hits and 5,244,911,616 bytes read. Both returned the same `READY` answer
as their cold runs. These checks did not include a host reboot or a throughput
benchmark. On this host, the MTP4 FP16-KV 148K profile with SSD offloading
needed `GPU_UTIL=0.965` to retain its advertised 148,480-token capacity;
this is a launcher override for that host, not a change to the shipped profile.
