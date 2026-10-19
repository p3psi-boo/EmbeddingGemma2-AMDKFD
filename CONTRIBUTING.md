# Contributing

## Build and CPU-only checks

```bash
nix develop -c scripts/build.sh
nix develop -c ctest --test-dir build --output-on-failure
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/public_repo.py check
```

Workflows are disabled. The local build cross-compiles both GPU code objects;
run the CPU checks above manually. No model,
GPU device, SSH credentials, or privileged runner is required. A successful
build does not establish hardware support for a new target.

Keep runtime changes separate from documentation/measurement changes. Preserve
CPU reference numerics, gate experimental optimizations, and record downstream
vendor changes in `vendor/PATCHES.md`. Include architecture/wave width and both
successful and failed cases when changing synchronization or queue lifetime.

## Manual HTTP regression

Start the GPU server using the README configuration. In a separate terminal,
start the CPU reference (the CPU backend does not acquire the GPU process lock):

```bash
MODEL=models/embeddinggemma-2-Q8_0.gguf PORT=42781 \
scripts/run.sh --backend cpu --threads 8
```

Then run:

```bash
python3 tests/http_parity.py
python3 tests/http_parity.py --long
python3 tests/http_parity.py --wide
```

Stop each server with Ctrl-C when finished. Run manual GPU operator tests only
when the GPU server is stopped. Do not run upstream fault/stress tests as part
of normal CI or deployment. Stop a hardware test sequence after the first new
GPU fault and retain local diagnostics; do not automatically reset or reboot.

## Benchmarks

Disable caches and per-dispatch logging. Use distinct inputs to avoid in-flight
coalescing, report warmups, retain per-request samples, and measure HTTP from the
same host. Use both orders for comparisons. Never derive a performance speedup
from a CPU reference running on another machine. See `docs/benchmarks.md`.

## Public artifacts

Use ignored `results/` for raw diagnostics and `benchmarks/` only for reviewed
numeric datasets. Do not add weights, credentials, deployment configuration,
host logs, raw traces, smaps, device dumps, or personal absolute paths.
See `docs/releasing.md` for the history-free export and privacy checks.

New project-owned contributions use WTFPL v2; changes within vendored components
retain those components' licenses. Preserve all upstream license notices.
