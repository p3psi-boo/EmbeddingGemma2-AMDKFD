#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
model=models/embeddinggemma-2-Q8_0.gguf
hash=6f1bd4ac6c5df7444f9cca7ca36cafe6cfa34cd6f49fefb1e0b4be8143aed8bc
mkdir -p models
if [[ ! -f "$model" ]]; then
  curl -fL --retry 3 -o "$model.partial" \
    https://huggingface.co/unsloth/embeddinggemma-2-GGUF/resolve/031f0d4b35536f69ab3509d4893c923264fcf253/embeddinggemma-2-Q8_0.gguf
  printf '%s  %s\n' "$hash" "$model.partial" | sha256sum -c -
  mv "$model.partial" "$model"
fi
printf '%s  %s\n' "$hash" "$model" | sha256sum -c -
