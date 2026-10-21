#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
# shellcheck source=../../launcher.sh
source "$ROOT/launcher.sh"

assert_eq() {
  local expected=$1
  local actual=$2
  local label=$3
  [[ "$actual" == "$expected" ]] || {
    printf '%s: expected %q, got %q\n' "$label" "$expected" "$actual" >&2
    return 1
  }
}

expect_failure() {
  local label=$1
  shift
  set +e
  ( parse_launcher_args "$@" ) >/dev/null 2>&1
  local rc=$?
  set -e
  (( rc != 0 )) || {
    printf '%s: unexpectedly accepted\n' "$label" >&2
    return 1
  }
}

reset_test_state() {
  unset MODEL_DIR SERVED_NAME MAX_MODEL_LEN MAX_NUM_SEQS MAX_BATCHED_TOKENS \
    GPU_UTIL TP_SIZE PP_SIZE CHAT_TEMPLATE_FILE SERVICE_SCOPE PORT KV_CACHE_DTYPE \
    FOREGROUND
  CONFIG_OVERRIDE_SOURCE=()
  CONFIG_OVERRIDE_UNSET=()
  NON_INTERACTIVE=0
  init_config_registry
}

reset_test_state
parse_launcher_args \
  --model /mnt/models/Qwen3.8-27B-FP8 \
  --served-model-name qwen3.8-27b \
  --max-model-len 32768 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --port 8080
assert_eq "/mnt/models/Qwen3.8-27B-FP8" "$MODEL_DIR" "--model maps to MODEL_DIR"
assert_eq "qwen3.8-27b" "$SERVED_NAME" "--served-model-name maps to SERVED_NAME"
assert_eq "32768" "$MAX_MODEL_LEN" "--max-model-len maps to MAX_MODEL_LEN"
assert_eq "2" "$MAX_NUM_SEQS" "--max-num-seqs maps to MAX_NUM_SEQS"
assert_eq "4096" "$MAX_BATCHED_TOKENS" "--max-num-batched-tokens maps to MAX_BATCHED_TOKENS"
assert_eq "0.9" "$GPU_UTIL" "--gpu-memory-utilization maps to GPU_UTIL"
assert_eq "2" "$TP_SIZE" "--tensor-parallel-size maps to TP_SIZE"
assert_eq "lan" "$SERVICE_SCOPE" "--host 0.0.0.0 maps to lan scope"
assert_eq "8080" "$PORT" "--port maps to PORT"
assert_eq 1 "$NON_INTERACTIVE" "vLLM CLI flags force non-interactive mode"
[[ -n "${CONFIG_OVERRIDE_SOURCE[MODEL_DIR]+x}" ]]

reset_test_state
parse_launcher_args /mnt/models/Qwen3.8-27B-FP8
assert_eq "/mnt/models/Qwen3.8-27B-FP8" "$MODEL_DIR" "positional model tag maps to MODEL_DIR"

reset_test_state
expect_failure "second positional model tag" /mnt/models/A /mnt/models/B

reset_test_state
parse_launcher_args --host localhost
assert_eq "local" "$SERVICE_SCOPE" "--host localhost maps to local scope"

reset_test_state
parse_launcher_args --host 127.0.0.1
assert_eq "local" "$SERVICE_SCOPE" "--host 127.0.0.1 maps to local scope"

reset_test_state
parse_launcher_args --host ::1
assert_eq "local" "$SERVICE_SCOPE" "--host ::1 maps to local scope"

reset_test_state
parse_launcher_args --host 192.168.1.10
assert_eq "lan" "$SERVICE_SCOPE" "specific interface address maps to lan scope"

reset_test_state
parse_launcher_args --model=/tmp/model0 --model=/tmp/model1
assert_eq "/tmp/model1" "$MODEL_DIR" "--model=VALUE maps to MODEL_DIR"

reset_test_state
parse_launcher_args --kv-cache-dtype int8_per_token_head
assert_eq "int8_per_token_head" "$KV_CACHE_DTYPE" "int8-per-token-head KV precision"

reset_test_state
parse_launcher_args --kv-cache-dtype auto
assert_eq "float16" "$KV_CACHE_DTYPE" "--kv-cache-dtype auto maps to float16"

reset_test_state
parse_launcher_args --chat-template /tmp/chat-template.j2
assert_eq "/tmp/chat-template.j2" "$CHAT_TEMPLATE_FILE" "--chat-template maps to CHAT_TEMPLATE_FILE"

reset_test_state
parse_launcher_args --pipeline-parallel-size 2 --tensor-parallel-size=2
assert_eq "2" "$PP_SIZE" "--pipeline-parallel-size maps to PP_SIZE"
assert_eq "2" "$TP_SIZE" "--tensor-parallel-size=VALUE maps to TP_SIZE"

reset_test_state
parse_launcher_args --foreground
assert_eq "1" "$FOREGROUND" "--foreground boolean flag"

expect_failure "q8_0 is not a vLLM-style KV value" --kv-cache-dtype q8_0
expect_failure "f16 is not a vLLM-style KV value" --kv-cache-dtype f16
expect_failure "unsupported KV precision" --kv-cache-dtype q4_0
expect_failure "gpu memory utilization zero" --gpu-memory-utilization 0
expect_failure "gpu memory utilization above one" --gpu-memory-utilization 1.5
expect_failure "gpu memory utilization is not numeric" --gpu-memory-utilization 0.9x
expect_failure "negative context length" --max-model-len -5
expect_failure "invalid context length" --max-model-len 32k
expect_failure "invalid batched token count" --max-num-batched-tokens 0
expect_failure "llama short options are not accepted" -m /mnt/models/x
expect_failure "llama alias options are not accepted" --alias x
expect_failure "llama cache-type options are not accepted" --cache-type-k q8_0
expect_failure "llama ngl options are not accepted" -ngl 99
expect_failure "unknown vLLM-style option" --tensor-split 1,2
expect_failure "--host requires a value" --host
expect_failure "--model requires a value" --model

gguf_file=$(mktemp --suffix .gguf)
trap 'rm -f "$gguf_file"' EXIT
reset_test_state
MODEL_DIR=$gguf_file
if prepare_runtime_defaults >/dev/null 2>&1; then
  echo "GGUF checkpoint unexpectedly accepted" >&2
  exit 1
fi

echo "launcher_vllm_cli_flags_ok"
