# Downstream changes

Original repositories, commits and licenses are recorded in `versions.json`.
The vendored tree is source, not an automatically updated dependency.

## libkfd

- Expose read-only PM4 command words and a non-transactional queue snapshot
  (read/write pointer, exception payload, CWSR location/type) for opt-in
  `EG2_KFD_DIAGNOSE=1`. No queue scheduling or CWSR allocation behavior changes.
- Add `LIBKFD_CWSR_BACKEND=auto|svm|gtt` for allocation-path isolation.
  `auto` retains upstream policy; `gtt` uses a mapped driver-managed GTT BO,
  not anonymous SVM or USERPTR. CWSR size, header, queue ioctls and destruction
  order remain the same. Linux 6.18.28 `kfd_queue_acquire_buffers` accepts this
  BO path before falling back to SVM range lookup. See [the public validation report](../docs/benchmarks.md).
- Add AQL diagnostic snapshots using the actual read/write dispatch IDs.
- CMake separates raw AMDGPU Clang from the Nix host wrapper. The wrapper adds
  x86 hardening flags that are invalid for AMDGPU.
- Build `gfx9-generic` and `gfx11-generic` trap handlers. Model code objects
  target `gfx90c` (wave64) and `gfx1103` (wave32) explicitly.
- Host C++ module scanning is disabled by the parent build (there are no modules).
- Fix `pm4::nop_fill`: split ring-tail padding into packets of at most 16384
  dwords; NOP COUNT=0x3fff means a one-dword NOP. The upstream implementation
  truncated large counts and filled the remaining stream with zeroes. A
  whole-graph submission exposed illegal PM4 opcodes on gfx90c. The regression
  test decodes 156 padding sizes on CPU; it never sends invalid packets to a GPU.
  Encoding reference: Linux `gfx_v9_ring_insert_nop` in
  [gfx_v9_0.c](https://github.com/gregkh/linux/blob/v6.18.28/drivers/gpu/drm/amd/amdgpu/gfx_v9_0.c).

## embeddinggemma.c

- Add opt-in `kfd` backend routing at the existing native backend boundary.
  Implementation lives in root `src/`, GPU kernels in root `kernels/`.
- `--text-only` skips multimodal projector download/initialization. KFD is
  text-only regardless of this flag.
- KFD defaults to a 2048-token packed-wave budget and one executor. Unsupported
  pipeline slots, CPU overflow and multi-device options fail explicitly.
- Optional strict input-token admission prevents silent truncation for KFD;
  over-limit inputs return HTTP 400 before queueing. Other backends preserve
  upstream behavior.

Upstream CPU graph and tokenizer numerical implementations are unchanged.

## Public-source packaging

- Remove machine-specific remote access details and personal absolute paths
  from vendored development notes; retain technical guidance and attribution.
- Preserve upstream licenses; install them alongside the project license.

## Dependency locking

- Pin optional libkfd Catch2 tests to the commit behind v3.13.0; use a full
  fetch so the commit remains reachable when upstream branches move.
- Pin the retained upstream CI checkout action and its Linux runner label.
- Pin the legacy 300M model URL to a Hugging Face revision (not a KFD target).
- Pin the retained upstream binary installer to v0.3.1 and the SHA-256 of its
  checksum manifest; reject release overrides that do not match this lock.
- Replace floating fetch/install examples with the pinned vendored equivalents.
  These optional upstream paths are not exercised by the KFD package build.

## Workflow removal

- Remove root and vendored GitHub Actions workflow files at maintainer request.
  Build and dependency-lock checks remain available for local execution.
