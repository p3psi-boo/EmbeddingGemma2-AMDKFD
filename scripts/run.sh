#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# The launcher explicitly opts into the validated fast path. Direct binary
# invocation keeps the baseline tile unless EG2_KFD_TILE=64 is set.
export EG2_KFD_TILE="${EG2_KFD_TILE:-64x64}"
binary="${EG2_BINARY:-./build/embeddinggemma-kfd}"
if [[ -z "${MODEL:-}" ]]; then
  scripts/fetch-model.sh
elif [[ ! -f "$MODEL" ]]; then
  printf 'Model file does not exist: %s\n' "$MODEL" >&2
  exit 1
fi
exec "$binary" --backend kfd --text-only \
  --model "${MODEL:-models/embeddinggemma-2-Q8_0.gguf}" \
  --bind "${BIND:-127.0.0.1}" --port "${PORT:-42780}" \
  --workers 4 --tokenizer-workers 2 --threads 2 \
  --max-batch-tokens 2048 --max-batch-sequence-tokens 2048 \
  --max-batch-requests 32 --batch-wait-us 200 \
  --cache-entries 0 --response-cache-mb 0 --media-cache-mb 0 "$@"
