# Architecture

```text
POST /v1/embeddings
  → native C HTTP frontend and tokenizer
  → dynamic packed batching, per-sequence boundaries
  → fixed EmbeddingGemma 2 text graph
  → Q8_0/BF16/F32 weights, FP32 activations and accumulation
  → libkfd memory allocation, PM4/AQL queues, doorbells and signals
  → Linux amdgpu/KFD → gfx90c or gfx1103
```

The HTTP frontend, GGUF reader, tokenizer, CPU reference, and batching service
come from pinned embeddinggemma.c sources. The KFD backend is in
`src/gemma2_kfd.cpp`; native operators and their ABI are in `kernels/`.
`vendor/PATCHES.md` records integration and libkfd changes.

## Execution and lifetime

- Quantized weights stay quantized in GPU-accessible memory; there is no full
  FP32 weight copy. A reusable arena stores activations and packed sequence data.
- One executor serializes GPU submissions. Packed ragged sequences preserve
  their own positions and bidirectional/sliding-window attention boundaries.
- Per-user process locking prevents this project's server and test tools from
  simultaneously owning GPU contexts. It does not lock out other applications
  or other users.
- Submission chunks contain 1–24 layers. Dependencies and cache barriers remain
  explicit; kernarg storage is reused only after completion.
- Queues are destroyed before their backing buffers and GPU context.
- The PM4 NOP padding encoder has CPU regression coverage at ring-tail/count
  boundaries. Opt-in structural packet validation checks lengths, not all
  opcode semantics or GPU memory access correctness.

## Queue and save-area choices

`LIBKFD_CWSR_BACKEND=gtt` allocates the compute wave save/restore area in a
mapped, driver-managed GTT buffer object rather than anonymous SVM. This is a
separate allocation choice from `EG2_KFD_QUEUE=pm4|aql`.

PM4 is the default model queue. AQL loads the code object using a temporary PM4
queue, waits, destroys that queue, and then creates the AQL execution queue.
Each AQL dispatch uses a barrier and system-scope acquire/release. A completion
signal at the end of the chunk gates host-side reuse. AQL currently does not
support the PM4 GPU timestamp profiler.

AMDHSA code objects and AQL packets are hardware/compiler ABIs. Using them here
does not load the ROCr/HSA runtime. The build generates `gfx90c` wave64 and
`gfx1103` wave32 code objects explicitly; device selection rejects other targets.

## Optimization switches

- `64`: padded LDS tile with four outputs per thread.
- `64x64`: larger output tile for at least 64 packed tokens; smaller inputs use
  the smaller tile.
- `step8`: 64×64 output tile with K-step 8, reducing LDS/VGPR pressure; shorter
  inputs retain the smaller kernel.
- Fusion combines normalization, residual/scale, and Q/K normalization with
  RoPE. Reduction order can differ between wave widths.

The CPU reference quantizes activations; the GPU path keeps FP32 activations.
CPU cosine parity is a regression criterion, not bitwise equality or an
independent full-precision model certification.
