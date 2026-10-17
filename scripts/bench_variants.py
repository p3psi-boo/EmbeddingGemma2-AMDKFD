#!/usr/bin/env python3
"""Paired, both-orders localhost HTTP experiment; owns/stops only its own server.

Run on the GPU host from the repository root. No clock/driver changes. Outputs
raw HTTP samples and exact settings; shared-host noise is not a speedup proof.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

CONFIGS = {
    "baseline": ("kfd", "64x64", "0", "1"),
    "step8": ("kfd", "step8", "0", "1"),
    "fusion": ("kfd", "step8", "1", "1"),
    "chunk": ("kfd", "step8", "0", "24"),
    "full": ("kfd", "step8", "1", "24"),
    "cpu": ("cpu", "64x64", "0", "1"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="+", choices=CONFIGS, default=["baseline", "step8", "fusion", "cpu"])
    ap.add_argument("--output", type=Path, default=Path("results/paired-round2"))
    ap.add_argument("--port", type=int, default=42780)
    ap.add_argument("--binary", default="build/embeddinggemma-kfd")
    ap.add_argument("--model", default="models/embeddinggemma-2-Q8_0.gguf")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for order, names in enumerate((args.configs, list(reversed(args.configs)))):
        for name in names:
            # Fail instead of accidentally benchmarking or terminating an existing service.
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", args.port))
            backend, tile, fuse, chunk = CONFIGS[name]
            settings = dict(EG2_KFD_TILE=tile, EG2_KFD_FUSE=fuse,
                            EG2_KFD_CHUNK_LAYERS=chunk, EG2_KFD_PROFILE="1")
            command = [args.binary, "--backend", backend, "--text-only", "--model", args.model,
                       "--bind", "127.0.0.1", "--port", str(args.port), "--workers", "4",
                       "--threads", "8" if backend == "cpu" else "2", "--tokenizer-workers", "2",
                       "--max-batch-tokens", "2048", "--max-batch-sequence-tokens", "2048",
                       "--max-batch-requests", "32", "--cache-entries", "0",
                       "--response-cache-mb", "0", "--media-cache-mb", "0"]
            with (args.output / f"{order}-{name}.log").open("w") as log:
                server = subprocess.Popen(command, env=os.environ | settings, stdout=log, stderr=log)
                try:
                    deadline = time.monotonic() + 60
                    while True:
                        if server.poll() is not None:
                            raise RuntimeError(f"{name} exited {server.returncode}; see {log.name}")
                        try:
                            with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/healthz", timeout=1):
                                break
                        except OSError:
                            if time.monotonic() > deadline:
                                raise TimeoutError(f"startup: {name}")
                            time.sleep(.1)
                    for batch, concurrency, requests in [(4, 1, 12), (4, 4, 16), (16, 1, 8)]:
                        bench = subprocess.run([sys.executable, "scripts/bench_http.py", "--port", str(args.port),
                            "--batch", str(batch), "--concurrency", str(concurrency), "--requests", str(requests),
                            "--label", name], check=False, capture_output=True, text=True, timeout=180)
                        if bench.returncode:
                            failure = dict(order=order, config=name, batch=batch, concurrency=concurrency,
                                           returncode=bench.returncode, stdout=bench.stdout, stderr=bench.stderr,
                                           settings=settings, command=command)
                            (args.output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
                            raise RuntimeError(f"{name} benchmark failed; see {args.output / 'failure.json'}")
                        row = json.loads(bench.stdout)
                        row.update(order=order, settings=settings, command=command)
                        results.append(row)
                        print(f"order={order} {name} b{batch}c{concurrency}: {row['tokens_per_second']:.1f} tok/s", flush=True)
                        (args.output / "http.json").write_text(json.dumps(results, indent=2) + "\n")
                finally:
                    server.terminate()
                    try:
                        server.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()
            time.sleep(2)


if __name__ == "__main__":
    main()
