# EmbeddingGemma 2 on `/dev/kfd`

**An experimental, OpenAI-compatible text embedding server for AMD APUs — directly
on Linux KFD, without a ROCm runtime.**

Native GPU kernels, quantized model weights, packed dynamic batching, and PM4/AQL
queue submission. No HIP, ROCr, rocBLAS, OCML, PyTorch, or llama.cpp runtime.
LLVM's AMDGPU backend compiles the kernels ahead of time.

[中文说明](docs/README.zh-CN.md) · [Architecture](docs/architecture.md) ·
[Benchmarks](docs/benchmarks.md) · [MES investigation](docs/mes.md) ·
[Contributing](CONTRIBUTING.md)

## Status

| Target | Evidence |
| --- | --- |
| `gfx1103`, wave32 | GTT CWSR + PM4 and AQL: kernel, HTTP parity, and bounded throughput tests passed |
| `gfx90c`, wave64 | Earlier text inference and parity passed; the newer GTT CWSR/AQL combinations have not been hardware-validated |
| Other architectures | Not supported by this model backend |

On one `gfx1103` engineering-sample APU, the measured peak in the tested matrix
was **3,167 tokens/s** (16 texts/request, HTTP concurrency 4, PM4). This is a
short, cache-off measurement, not a production throughput guarantee or a
same-host CPU speedup claim. See [methodology and samples](docs/benchmarks.md).

This is a research backend, not a production-qualified service. Earlier runs
encountered GPU resets and MES queue-removal failures. GTT-backed CWSR avoided
the observed failure in subsequent tests; long-duration stability and
independent model-reference parity remain open work.

## Build

Requirements: x86-64 Linux, Nix with `nix-command` and `flakes` enabled. Running
the GPU backend additionally requires the amdgpu/KFD driver and access to
`/dev/kfd` and the relevant DRM render node. Compilation and CPU tests need no GPU.

```bash
nix develop -c scripts/build.sh
nix develop -c ctest --test-dir build --output-on-failure
python3 -m unittest discover -s tests -p 'test_*.py'

# Alternative: a complete package, linked at ./result
nix build --cores 8
```

`flake.lock` pins the toolchain. The build uses Clang 21, CMake, Ninja, and
libdrm; GPU compilation uses raw Clang rather than the host compiler wrapper.
Both supported code objects are built. For an unavailable locally configured
Nix mirror, append `--option substituters https://cache.nixos.org`.

## Model

```bash
scripts/fetch-model.sh
```

The script downloads `embeddinggemma-2-Q8_0.gguf` from
[unsloth/embeddinggemma-2-GGUF](https://huggingface.co/unsloth/embeddinggemma-2-GGUF/tree/031f0d4b35536f69ab3509d4893c923264fcf253),
pinned to revision `031f0d4b35536f69ab3509d4893c923264fcf253`, and checks SHA-256:

```text
6f1bd4ac6c5df7444f9cca7ca36cafe6cfa34cd6f49fefb1e0b4be8143aed8bc
```

Weights are downloaded separately and are not included in this repository.
This backend supports the text graph of this Q8_0 model, not arbitrary GGUFs.

## Run on gfx1103

The following explicitly selects the configuration tested with GTT-backed CWSR.
Library defaults remain unchanged; running `scripts/run.sh` without these
variables does **not** select this configuration.

```bash
LIBKFD_CWSR_BACKEND=gtt EG2_KFD_QUEUE=pm4 \
EG2_KFD_TILE=step8 EG2_KFD_FUSE=1 EG2_KFD_CHUNK_LAYERS=24 EG2_KFD_PROFILE=0 \
scripts/run.sh
```

For a Nix package, also set `EG2_BINARY=./result/bin/embeddinggemma-kfd`.
Set `MODEL` to use an existing model file. The launcher binds only to
`127.0.0.1:42780` by default and disables caches for reproducible measurements.
There is no built-in authentication or TLS. Network exposure belongs behind an
authenticated reverse proxy; see [Security](SECURITY.md).

```bash
curl http://127.0.0.1:42780/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"embeddinggemma-2","input":["hello world","你好"],"dimensions":768,"encoding_format":"float"}'
```

### API contract

- `POST /v1/embeddings`: string or string-array input; output order matches input.
- `dimensions`: 128, 256, 512, or 768; embeddings are L2-normalized after truncation.
- `encoding_format`: `float` or `base64`; responses include token usage.
- `GET /healthz`: HTTP health endpoint, not proof of GPU health under future load.
- Maximum **2,048 tokens per sequence and per packed GPU wave**, including special
  tokens. Over-limit sequences return HTTP 400 instead of silent truncation.
  An HTTP batch may span multiple waves. This is a backend limit, not the model's
  full context capacity.
- Text-only, one GPU executor. No multimodal or token-ID-array inputs.
- Query/document instruction prefixes are supplied by the caller, not added automatically.

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `LIBKFD_CWSR_BACKEND` | `auto` | `auto`, `svm`, or driver-managed `gtt` queue save area |
| `EG2_KFD_QUEUE` | `pm4` | `pm4` or `aql`; AQL requires profiling off |
| `EG2_KFD_TILE` | launcher: `64x64`; binary: baseline | `16`, `64`, `64x64`, or `step8` GEMM selection |
| `EG2_KFD_FUSE` | `0` | Enable fused normalization/residual/RoPE operations with `1` |
| `EG2_KFD_CHUNK_LAYERS` | `1` | Submission granularity, 1–24 layers |
| `EG2_KFD_PROFILE` | `0` | PM4 GPU timestamp profiling with `1` |
| `EG2_KFD_DIAGNOSE` | `0` | Queue/fence diagnostics with `1`; not for throughput measurement |

The launcher also accepts `BIND`, `PORT`, `MODEL`, `EG2_BINARY`, and trailing server
arguments. Model weights and GPU workspace use approximately 474 MiB together;
HTTP, tokenizer, mappings, and other process memory are additional.

## Validation

CPU-only checks above run in CI. GPU checks are manual and should run one at a
time, with the server stopped:

```bash
LIBKFD_CWSR_BACKEND=gtt timeout 30s build/kfd-smoke build/eg2-gfx1103.elf
LIBKFD_CWSR_BACKEND=gtt timeout 60s build/kfd-kernel-test build/eg2-gfx1103.elf
LIBKFD_CWSR_BACKEND=gtt timeout 60s build/kfd-kernel-test build/eg2-gfx1103.elf --aql
```

HTTP parity, concurrent-batch validation, and cache-off throughput procedures are
in [Contributing](CONTRIBUTING.md). Diagnostic scripts never request a reboot or
GPU reset. A driver may nevertheless reset a faulting GPU automatically.

## Repository map

- `src/`: model backend, code-object selection, queue checks, process lock.
- `kernels/`: native AMDGPU operators and shared ABI.
- `tests/`: CPU regressions, GPU operator tests, HTTP parity.
- `scripts/`: build, model download, launch, benchmark, diagnostics, release checks.
- `benchmarks/`: reviewed hardware metadata and numeric samples only.
- `vendor/`: pinned sources, original licenses, and documented downstream changes.

Raw traces, memory dumps, host logs, weights, local build outputs, and deployment
configuration are excluded from the public source tree.

## License

Project-owned code is licensed under **[WTFPL v2](LICENSE)**. Vendored code retains
its original licenses: libkfd under Apache-2.0 and embeddinggemma.c under MIT.
Model weights have their own upstream license. See [third-party notices](THIRD_PARTY_NOTICES.md).
