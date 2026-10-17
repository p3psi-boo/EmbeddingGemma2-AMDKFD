# MES queue-removal investigation

This report separates observed evidence from hypotheses. It intentionally
contains no machine identifiers, raw memory addresses, or device dumps.

## Observed failure

On gfx1103 with anonymous-SVM CWSR, a single-process run using `step8`, fusion,
and one-layer chunks reproduced this sequence:

```text
kcompactd / memory compaction
  → anonymous page migration
  → svm_range_cpu_invalidate_pagetables
  → kgd2kfd_quiesce_mm
  → process queue eviction
  → MES REMOVE_QUEUE timeout (approximately 2 seconds)
  → automatic driver GPU reset
  → application memory_lost exception
```

There were 1,385 submissions and 1,384 completed fence checks. Eight earlier
ring wraps completed; the last submission was 1,637 DWORDs and did not wrap.
Queue destruction appeared after the fault, not as its initiating event.

The failure recurred after project-wide process exclusion was added. Overlapping
tools and whole-graph submission therefore are not necessary conditions. PM4
structural validation and good previous fences narrow the investigation but do
not prove all kernel memory accesses or hardware execution correct.

CWSR was the library's anonymous-SVM allocation and is the leading candidate for
the invalidated range. The trace did not record the notifier range address, so
it did not directly identify the exact page. The evidence does not independently
assign the underlying timeout to a firmware-only defect.

## GTT CWSR comparison

The added `LIBKFD_CWSR_BACKEND=gtt` path uses a mapped driver-managed GTT BO,
keeping the CWSR size, header and queue ABI unchanged. It does not relabel
anonymous memory as USERPTR or depend on `mlock` to prevent migration.
Linux 6.18.28's
[`kfd_queue_acquire_buffers`](https://github.com/gregkh/linux/blob/v6.18.28/drivers/gpu/drm/amd/amdkfd/kfd_queue.c)
accepts a BO-backed queue buffer before falling back to SVM-range lookup.

GTT + PM4 and GTT + AQL subsequently passed kernel/HTTP tests and the bounded
throughput matrix described in [Benchmarks](benchmarks.md). No new MES error or
reset was observed in those runs, while system compaction/migration continued.
This avoids the observed anonymous-SVM invalidation trigger; it is not a MES
firmware fix or a production-stability certification. AQL + SVM and the new
GTT combinations on gfx90c remain untested.

## Diagnostic tools

- `EG2_KFD_DIAGNOSE=1`: packet-length checks and queue/fence snapshots.
- `scripts/diagnose_mes.py`: bounded sequential test matrix; stops after the
  first detected new kernel GPU fault. Requires kernel journal access.
- `scripts/trace_mes.sh`: owns a separate tracefs instance and captures a bounded
  partial device dump. Uses noninteractive sudo for tracing/journal access.
- `scripts/analyze_mes.py`: offline analysis of the layer-run diagnostic layout.

Use a new directory under ignored `results/` for captures. These tools run GPU
work and can capture hostnames, command lines, paths, memory contents and kernel
addresses. Keep the raw outputs local; publish reviewed summaries and numeric
samples instead. Do not attach device dumps to public issues.

The tools do not request reboots, GPU resets, driver reloads, or changes to
system-wide THP/NUMA/compaction settings. A GPU fault can still cause an automatic
driver reset. The investigated run did; the later GTT runs did not.
