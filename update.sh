#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_RELEASE_FILE=${PROJECT_RELEASE_FILE:-"$ROOT/PROJECT_RELEASE.env"}
# shellcheck source=/dev/null
source "$PROJECT_RELEASE_FILE"

REMOTE_REPO=${REMOTE_REPO:-https://github.com/Godl1nk/vLLM-2080Ti-Definitive}
REMOTE_RELEASE_API=${REMOTE_RELEASE_API:-https://api.github.com/repos/Godl1nk/vLLM-2080Ti-Definitive/releases/latest}
UPDATE_PROMPT_TIMEOUT_SECONDS=${UPDATE_PROMPT_TIMEOUT_SECONDS:-10}
UPDATE_BUILD_PROMPT_TIMEOUT_SECONDS=${UPDATE_BUILD_PROMPT_TIMEOUT_SECONDS:-10}
UPDATE_DOWNLOAD_TIMEOUT_SECONDS=${UPDATE_DOWNLOAD_TIMEOUT_SECONDS:-60}

# Runtime state is local to a host and must survive a release refresh. Tracked
# route profiles are replaced by the release archive; profiles/local is for
# site-specific presets that are intentionally outside the shipped matrix.
PRESERVE_PATHS=(
  ".venv"
  ".deps"
  "build-logs"
  "run-logs"
  "results"
  "torchinductor-cache"
  "triton-cache"
  ".omx"
  "backups"
  "profiles/local"
)

prompt_yes_no_timeout() {
  local prompt=$1 timeout_seconds=${2:-10} default_answer=${3:-y} answer=""
  if [[ "${ASSUME_YES:-0}" == "1" || "${YES:-0}" == "1" || ! -t 0 ]]; then
    printf '%s\n' "$default_answer"
    return 0
  fi
  printf '%s ' "$prompt" >&2
  if read -r -t "$timeout_seconds" answer; then
    case "$answer" in
      n|N|no|NO) printf 'n\n' ;;
      *) printf '%s\n' "$default_answer" ;;
    esac
  else
    printf '%s\n' "$default_answer"
  fi
}

latest_remote_version() {
  python3 - "$REMOTE_RELEASE_API" <<'PY'
import json
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=10) as response:
        tag = json.load(response).get("tag_name", "").strip()
    if tag.startswith("v"):
        tag = tag[1:]
    print(tag)
except Exception:
    pass
PY
}

read_local_version() {
  tr -d '[:space:]' < "$ROOT/VERSION"
}

is_remote_newer() {
  local local_version=$1 remote_version=$2
  [[ -n "$remote_version" && "$remote_version" != "$local_version" ]] || return 1
  [[ "$(printf '%s\n%s\n' "$local_version" "$remote_version" | sort -V | tail -n 1)" == "$remote_version" ]]
}

download_archive() {
  local version=$1 tmpdir archive_path strip_root
  tmpdir=$(mktemp -d)
  trap 'rm -rf "$tmpdir"' RETURN
  archive_path="$tmpdir/src.tar.gz"
  strip_root="vLLM-2080Ti-Definitive-${version}"

  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --max-time "$UPDATE_DOWNLOAD_TIMEOUT_SECONDS" \
      -o "$archive_path" "${REMOTE_REPO%/}/archive/refs/tags/v${version}.tar.gz"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$archive_path" --timeout="$UPDATE_DOWNLOAD_TIMEOUT_SECONDS" \
      "${REMOTE_REPO%/}/archive/refs/tags/v${version}.tar.gz"
  else
    echo "Neither curl nor wget was found." >&2
    return 1
  fi
  tar -xzf "$archive_path" -C "$tmpdir"

  for path in "${PRESERVE_PATHS[@]}"; do
    [[ -e "$ROOT/$path" ]] || continue
    mkdir -p "$tmpdir/preserve/$(dirname -- "$path")"
    cp -a "$ROOT/$path" "$tmpdir/preserve/$path"
  done

  find "$ROOT" -mindepth 1 -maxdepth 1 \
    \( -name .git -o -name .venv -o -name .deps -o -name build-logs \
       -o -name run-logs -o -name results -o -name torchinductor-cache \
       -o -name triton-cache -o -name .omx -o -name backups \
       -o -path "$ROOT/profiles/local" \) -prune -o -exec rm -rf {} +
  cp -a "$tmpdir/$strip_root"/. "$ROOT"/

  for path in "${PRESERVE_PATHS[@]}"; do
    [[ -e "$tmpdir/preserve/$path" ]] || continue
    mkdir -p "$ROOT/$(dirname -- "$path")"
    cp -a "$tmpdir/preserve/$path" "$ROOT/$path"
  done
}

main() {
  echo "$PROJECT_NAME v$FORK_RELEASE update helper"
  local local_version remote_version
  local_version=$(read_local_version)
  remote_version=$(latest_remote_version || true)
  echo "Local version:  ${local_version:-unknown}"
  echo "Remote version: ${remote_version:-unavailable}"

  if ! is_remote_newer "$local_version" "$remote_version"; then
    echo "Already up to date."
    return 0
  fi

  echo "New release available: v$remote_version"
  [[ "$(prompt_yes_no_timeout "Update now? [Y/n]:" "$UPDATE_PROMPT_TIMEOUT_SECONDS" y)" != n ]] || return 0
  echo "Updating from release archive v$remote_version..."
  download_archive "$remote_version"
  echo "Update complete. Current version: $(read_local_version)"

  if [[ "$(prompt_yes_no_timeout "Build now? [Y/n]:" "$UPDATE_BUILD_PROMPT_TIMEOUT_SECONDS" y)" != n ]]; then
    exec bash "$ROOT/build.sh"
  fi
}

main "$@"
