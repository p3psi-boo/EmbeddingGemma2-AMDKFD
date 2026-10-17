#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLIBKFD_GPU_ARCHS="gfx9-generic;gfx11-generic" \
  -DLIBKFD_GPU_COMPILER="${KFD_GPU_COMPILER:?Run inside nix develop}" \
  -DLIBKFD_GPU_RESOURCE_DIR="${KFD_GPU_RESOURCE_DIR:?Run inside nix develop}"
cmake --build build -j "${JOBS:-8}"
