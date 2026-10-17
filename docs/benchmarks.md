# Measurements and validation

## gfx1103 / GTT CWSR — 2026-10-17

The reviewed dataset is in [`benchmarks/gfx1103-gtt`](../benchmarks/gfx1103-gtt).
Hardware metadata intentionally omits hostnames, addresses, usernames, serial
numbers, deployment paths, and kernel memory addresses.

- APU: AMD engineering sample `100-000000964-50_Y`, `gfx1103`, wave32, 24 SIMDs.
  This is not identified as a specific retail CPU model.
- System RAM: approximately 30 GiB; reported VRAM: 1 GiB.
- Linux 6.18.28; MES firmware `0x86`, MES KIQ firmware `0x109`.
- Q8_0 weights, FP32 activations; `step8`, fusion on, 24-layer chunks, GTT CWSR.
- Localhost HTTP, base64 output, all launcher caches off, GPU profiling and
  per-submission diagnostics off. Sparse function tracing remained enabled.
- Each cell: three disjoint warmups, 128 measured requests, 30 repeated words
  plus unique request/item prefixes (about 40 tokens per text).
- PM4 was measured before AQL, without reversed-order pairing. Other host work
  and thermal effects were not controlled sufficiently for a universal queue ranking.

| Texts/request | HTTP concurrency | PM4 tokens/s | AQL tokens/s | PM4 p95 ms | AQL p95 ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 2300 | 2336 | 73.6 | 71.9 |
| 4 | 4 | 2740 | 2771 | 244.8 | 252.4 |
| 16 | 1 | 2970 | 2808 | 238.0 | 251.8 |
| 16 | 4 | 3167 | 2960 | 890.8 | 935.1 |

Request latency includes queueing and response handling. All eight cells total
1,024 measured requests plus 24 warmups. These runs lasted seconds per cell,
not hours. No new MES error or GPU reset was observed in this matrix. There is
no new same-host CPU throughput measurement for this configuration.

### Accuracy and workload coverage

- PM4 and AQL each passed the same 74 kernel cases; maximum reported absolute
  error was `2.38418579e-7`.
- Layer-at-a-time and full-graph HTTP runs covered short multilingual inputs,
  a 1,038-token mixed batch, dimensions 128/256/512/768, float/base64, ordering,
  concurrency, invalid inputs, and health after invalid requests.
- Minimum CPU-reference cosine: `0.9998687901875911`. CPU reference timing is
  from a different machine and is not used for a speedup ratio.
- Distinct wide concurrent requests matched their separately executed GPU
  references with cosine 1.0. The observed packed workload reached 1,378 tokens
  and 32 sequences; this is not coverage of the full 2,048-token boundary.
- An earlier wide test used identical requests, which could be coalesced in
  flight even with caches off. The published wide regression uses four distinct
  prefixes. Throughput requests already used distinct request IDs.

System-level counters showed page migration and compaction during GTT tests.
They do not establish that any particular application buffer migrated. See
[the MES investigation](mes.md) for the prior failure and scope of the mitigation.

### Data provenance

`*-b*-c*.json` retain the numeric benchmark results and per-request timing arrays
from `scripts/bench_http.py`. `*-parity-*.json` contain selected HTTP regression
results. They are measurements from the pre-publication implementation; source
cleanup did not rerun GPU tests. `metadata.json` records the configuration.
Raw ftrace, smaps, device dumps, server commands, and host logs are private local
artifacts, not included in this public dataset. Documentation is a summarized
account of that diagnostic evidence, not a substitute for publishing raw traces.

## Earlier gfx90c results and failures

On a Ryzen 7 5700G (`gfx90c`), the earlier 64×64 path measured about 1,256 tokens/s
at batch 16 / concurrency 1, versus about 1,485 tokens/s on the same host's
8-thread CPU path. GPU timestamp profiling was on. This predates the GTT CWSR
work and is not comparable to the gfx1103 table above.

A large PM4 ring-tail submission exposed an upstream NOP-padding bug and caused
an illegal-opcode fault/reset. The encoder was corrected and 156 CPU padding
boundary cases added. Queue creation remained problematic after recovery;
subsequent gfx90c hardware validation was not performed. A successful CPU test
or cross-build is not a claim that the newer gfx90c runtime path was validated.

## Reproduction and next gates

Use the explicit launch configuration in the root README, then run, on the
server's host:

```bash
python3 scripts/bench_http.py --batch 16 --concurrency 4 --requests 128
```

Restart with `EG2_KFD_QUEUE=aql` for the AQL configuration. Do not overlap GPU
servers. Publish configuration, all samples, failures, and both measurement
orders. Next gates are independent model-reference parity, full token-boundary
coverage, repeated interleaved same-host CPU/GPU comparisons, and long soak tests
with latency distributions and fault monitoring.
