#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
# shellcheck source=../../launcher.sh
source "$ROOT/launcher.sh"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

mkdir -p "$tmp/bin" "$tmp/vision" "$tmp/text" "$tmp/int4"
printf '%s\n' \
  '#!/usr/bin/env bash' \
  'printf "0, NVIDIA GeForce RTX 2080 Ti\n"' \
  'printf "1, NVIDIA GeForce RTX 2080 Ti\n"' > "$tmp/bin/nvidia-smi"
chmod +x "$tmp/bin/nvidia-smi"
export PATH="$tmp/bin:$PATH"

fp8_config='{"quant_method":"compressed-tensors","config_groups":{"group_0":{"weights":{"num_bits":8,"type":"float"},"input_activations":{"num_bits":8,"type":"float"}}}}'
int4_config='{"quant_method":"compressed-tensors","config_groups":{"group_0":{"weights":{"num_bits":4,"type":"int"}}}}'
printf '%s\n' \
  "{\"model_type\":\"qwen3_5\",\"architectures\":[\"Qwen3_5VLForConditionalGeneration\"],\"vision_config\":{\"depth\":27},\"quantization_config\":$fp8_config,\"max_position_embeddings\":262144}" > "$tmp/vision/config.json"
printf '%s\n' \
  "{\"model_type\":\"qwen3_5\",\"architectures\":[\"Qwen3_5ForCausalLM\"],\"quantization_config\":$fp8_config,\"max_position_embeddings\":262144}" > "$tmp/text/config.json"
printf '%s\n' \
  "{\"model_type\":\"qwen3_5\",\"architectures\":[\"Qwen3_5ForCausalLM\"],\"quantization_config\":$int4_config,\"max_position_embeddings\":262144}" > "$tmp/int4/config.json"

[[ "$(guess_quantization "$tmp/vision")" == "fp8" ]]
[[ "$(guess_quantization "$tmp/int4")" == "compressed-tensors" ]]

EXPECTED_VISION="$ROOT/profiles/2x2080Ti/qwen27b/w8a16/mtp4-fp8kv-1x186K-text-image.env"
EXPECTED_TEXT="$ROOT/profiles/2x2080Ti/qwen27b/w8a16/mtp4-fp8kv-1x262K-text-only.env"
EXPECTED_INT4="$ROOT/profiles/2x2080Ti/qwen27b/w4a16/mtp4-fp8kv-2x229K-text-only.env"
[[ -f "$EXPECTED_VISION" && -f "$EXPECTED_TEXT" && -f "$EXPECTED_INT4" ]]

MODEL_DIR="$tmp/vision"
[[ "$(resolve_profile_file)" == "$EXPECTED_VISION" ]]

MODEL_DIR="$tmp/text"
[[ "$(resolve_profile_file)" == "$EXPECTED_TEXT" ]]

MODEL_DIR="$tmp/int4"
[[ "$(resolve_profile_file)" == "$EXPECTED_INT4" ]]

PROFILE=2x2080Ti/qwen27b/w8a16/mtp4-fp8kv-1x186K-text-image.env
MODEL_DIR="$tmp/text"
[[ "$(resolve_profile_file)" == "$EXPECTED_VISION" ]]
unset PROFILE

printf '%s\n' \
  '#!/usr/bin/env bash' \
  'printf "0, Tesla V100-SXM2-16GB\n"' \
  'printf "1, Tesla V100-SXM2-16GB\n"' > "$tmp/bin/nvidia-smi"
MODEL_DIR="$tmp/text"
if resolve_profile_file >/dev/null; then
  echo "expected no auto profile for unsupported hardware" >&2
  exit 1
fi

echo "launcher_auto_profile_ok"
