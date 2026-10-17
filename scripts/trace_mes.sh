#!/usr/bin/env bash
# Run as the normal GPU user; only tracing and journal access use sudo.
set -euo pipefail
cd "$(dirname "$0")/.."
out="${1:?usage: trace_mes.sh NEW_OUTPUT_DIRECTORY [diagnose_mes.py options]}"
shift
mkdir "$out"
out="$(realpath "$out")"
export EG2_TRACE_INSTANCE="/sys/kernel/tracing/instances/eg2-mes-$$"
cleanup() {
  sudo -n bash -s -- "$EG2_TRACE_INSTANCE" "$out" <<'ROOT'
set -eu
instance="$1"
[ -d "$instance" ] || exit 0
echo 0 > "$instance/tracing_on"
cat "$instance/trace" > "$2/ftrace.txt"
cat "$instance"/per_cpu/cpu*/stats > "$2/ftrace-stats.txt"
echo nop > "$instance/current_tracer"
rmdir "$instance"
for dev in /sys/class/devcoredump/devcd*/data; do
  [ -r "$dev" ] || continue
  # Read only: never acknowledge/delete a dump. Bound disk use to 64 MiB.
  name="$2/devcoredump-$(basename "$(dirname "$dev")")"
  status=0
  timeout -k 2s 15s head -c 67108864 "$dev" > "$name.partial.bin" || status=$?
  printf 'reader_exit=%s; bounded to 15 seconds / 64 MiB; completeness not assumed\n' "$status" > "$name.status.txt"
done
ROOT
}
trap cleanup EXIT
sudo -n bash -s -- "$EG2_TRACE_INSTANCE" <<'ROOT'
set -eu
mkdir "$1"
echo 0 > "$1/tracing_on"
echo mono > "$1/trace_clock"
echo 2048 > "$1/buffer_size_kb"
grep -E '^(kgd2kfd_quiesce_mm|kfd_ioctl_destroy_queue|kfd_ioctl_create_queue|kfd_hws_hang|remove_queue_mes|evict_process_queues_cpsch|svm_range_cpu_invalidate_pagetables|amdgpu_amdkfd_evict_userptr|amdgpu_amdkfd_gpuvm_alloc_memory_of_gpu|amdgpu_amdkfd_gpuvm_free_memory_of_gpu)([. ]|$)' /sys/kernel/tracing/available_filter_functions | awk '{print $1}' > "$1/set_ftrace_filter"
echo function > "$1/current_tracer"
echo 1 > "$1/options/func_stack_trace"
echo 1 > "$1/tracing_on"
ROOT
timeout -s TERM -k 20s 420s python3 scripts/diagnose_mes.py --output "$out" "$@"
