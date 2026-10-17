#!/usr/bin/env python3
"""Bounded, sequential queue diagnostics. Run on the GPU host, not over HTTP SSH.

Stops the matrix on the first new kernel fault. Does not reset hardware, change
driver options, or create concurrent project GPU processes. Trace wrapper owns
an independent tracefs instance; diagnostic runs are not performance results.
"""
import argparse
import http.client
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

FAULT = re.compile(r"MES failed|GPU reset begin|Failed to evict|GPU fault|page fault|illegal opcode", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--binary", default="build/embeddinggemma-kfd")
    ap.add_argument("--cwsr", choices=["auto", "svm", "gtt"], default="auto")
    ap.add_argument("--requests", type=int, default=24)
    ap.add_argument("--queue", choices=["pm4", "aql"], default="pm4")
    ap.add_argument("--cpu-reference", help="Run short and long HTTP parity against this CPU server URL")
    ap.add_argument("--parity-modes", nargs="+", choices=["short", "long", "wide"], default=["short", "long", "wide"])
    ap.add_argument("--wide-batch", action="store_true", help="Also exercise 16 sequences/request at concurrency 4")
    ap.add_argument("--no-diagnostics", action="store_true", help="Disable per-submission stderr for timing runs")
    ap.add_argument("--variants", nargs="+", choices=["layer", "full"], default=["layer", "full"])
    args = ap.parse_args()
    if not 1 <= args.requests <= 512:
        ap.error("--requests must be 1..512")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out/"config.json").write_text(json.dumps(vars(args), indent=2)+"\n")
    (out/"vmstat-before.txt").write_text(Path("/proc/vmstat").read_text())
    start = str(int(time.time()))
    rows = []

    def kernel_log():
        return subprocess.check_output(["sudo", "-n", "journalctl", "-k", "--since", "@"+start,
                                        "--no-pager", "-o", "short-monotonic"], text=True)

    def healthy():
        log = kernel_log()
        (out/"kernel.log").write_text(log)
        if FAULT.search(log):
            raise RuntimeError("new kernel GPU fault; remaining GPU tests cancelled")

    def marker(message):
        # Use CLOCK_MONOTONIC like ftrace; avoid trace_marker, whose user-copy
        # path returned EFAULT/WARN on this host before any GPU test launched.
        with (out/"markers.jsonl").open("a") as f:
            f.write(json.dumps(dict(ns=time.monotonic_ns(), message=message))+"\n")

    def interrupted(signum, frame):
        raise RuntimeError(f"diagnostic interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)

    def run_checked(command, label):
        with (out/(label+".json")).open("w") as result, (out/(label+".err")).open("w") as err:
            child = subprocess.Popen(command, stdout=result, stderr=err)
            try:
                deadline=time.monotonic()+120
                while child.poll() is None:
                    healthy()
                    if time.monotonic()>deadline:
                        raise TimeoutError(label)
                    time.sleep(.5)
                if child.returncode:
                    raise RuntimeError(f"test failed: {label}")
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=10)
        healthy()
        rows.append(dict(test=label, status="pass"))
        print(label+": pass", flush=True)

    try:
        healthy()
        for variant in args.variants:
            marker("start "+variant)
            env = dict(os.environ, EG2_KFD_TILE="step8", EG2_KFD_FUSE="1",
                       EG2_KFD_CHUNK_LAYERS="1" if variant=="layer" else "24",
                       EG2_KFD_DIAGNOSE="0" if args.no_diagnostics else "1", EG2_KFD_PROFILE="0",
                       LIBKFD_CWSR_BACKEND=args.cwsr, EG2_KFD_QUEUE=args.queue)
            cmd = [args.binary, "--backend", "kfd", "--text-only", "--model",
                   "models/embeddinggemma-2-Q8_0.gguf", "--bind", "127.0.0.1", "--port", "42780",
                   "--workers", "4", "--threads", "2", "--tokenizer-workers", "2",
                   "--max-batch-tokens", "2048", "--max-batch-sequence-tokens", "2048",
                   "--max-batch-requests", "32", "--cache-entries", "0",
                   "--response-cache-mb", "0", "--media-cache-mb", "0"]
            with (out/(variant+".log")).open("w") as log:
                server = subprocess.Popen(cmd, env=env, stdout=log, stderr=log, start_new_session=True)
                try:
                    deadline = time.monotonic()+90
                    while True:
                        healthy()
                        if server.poll() is not None:
                            raise RuntimeError(f"server exited during startup: {server.returncode}")
                        conn = http.client.HTTPConnection("127.0.0.1", 42780, timeout=1)
                        try:
                            conn.request("GET", "/healthz")
                            if conn.getresponse().status == 200:
                                break
                        except OSError:
                            pass
                        finally:
                            conn.close()
                        if time.monotonic()>deadline:
                            raise TimeoutError("server startup")
                        time.sleep(.5)
                    (out/(variant+"-smaps.txt")).write_text(Path(f"/proc/{server.pid}/smaps").read_text())
                    shapes=[(4, 1), (4, 4), (16, 1)] + ([(16, 4)] if args.wide_batch else [])
                    for batch, concurrency in shapes:
                        label=f"{variant}-b{batch}-c{concurrency}"
                        marker(label)
                        run_checked(["python3", "scripts/bench_http.py", "--batch", str(batch),
                            "--concurrency", str(concurrency), "--requests", str(args.requests), "--label", label],label)
                    if args.cpu_reference:
                        for mode in args.parity_modes:
                            label=variant+"-parity-"+mode
                            marker(label)
                            run_checked(["python3", "tests/http_parity.py", "--cpu", args.cpu_reference]
                                        + (["--"+mode] if mode!="short" else []),label)
                    marker("idle "+variant)
                    time.sleep(2)
                    healthy()
                finally:
                    marker("stop "+variant)
                    if server.poll() is None:
                        os.killpg(server.pid, signal.SIGTERM)
                        try:
                            server.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(server.pid, signal.SIGKILL)
                            server.wait(timeout=10)
                    rows.append(dict(test=variant+"-server-exit", returncode=server.returncode))
            time.sleep(2)
            healthy()
    except Exception as exc:
        rows.append(dict(status="failed", error=str(exc)))
        print(str(exc), flush=True)
        raise
    finally:
        (out/"summary.json").write_text(json.dumps(rows, indent=2)+"\n")
        (out/"kernel.log").write_text(kernel_log())
        (out/"vmstat-after.txt").write_text(Path("/proc/vmstat").read_text())


if __name__ == "__main__":
    main()
