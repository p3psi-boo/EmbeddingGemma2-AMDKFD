#!/usr/bin/env python3
"""Cache-off HTTP throughput measurement; run on the server host to exclude SSH latency."""
import argparse
import concurrent.futures
import http.client
import json
import math
import statistics
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=42780)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--words", type=int, default=30)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--requests", type=int, default=12)
    ap.add_argument("--label", default="kfd-tile64")
    args = ap.parse_args()

    def run(i):
        payload = dict(model="embeddinggemma-2", encoding_format="base64",
                       input=[f"benchmark {i} item {j}: " + "word "*args.words for j in range(args.batch)])
        conn = http.client.HTTPConnection("127.0.0.1", args.port, timeout=180)
        start = time.perf_counter()
        try:
            conn.request("POST", "/v1/embeddings", json.dumps(payload), {"Content-Type": "application/json"})
            response = conn.getresponse()
            body = response.read()
            if response.status != 200:
                raise RuntimeError((response.status, body[:300]))
            result = json.loads(body)
            assert len(result["data"]) == args.batch
            return result["usage"]["total_tokens"], time.perf_counter()-start
        finally:
            conn.close()

    # Disjoint warmup inputs; both server caches must be explicitly disabled.
    for i in range(3):
        run(10000+i)
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(run, range(args.requests)))
    elapsed = time.perf_counter()-start
    latencies = sorted(row[1] for row in rows)
    print(json.dumps(dict(label=args.label, batch=args.batch, words=args.words,
                         concurrency=args.concurrency, requests=args.requests,
                         elapsed_seconds=elapsed, tokens=sum(r[0] for r in rows),
                         tokens_per_second=sum(r[0] for r in rows)/elapsed,
                         sequences_per_second=args.requests*args.batch/elapsed,
                         requests_per_second=args.requests/elapsed,
                         median_request_ms=statistics.median(latencies)*1000,
                         p95_request_ms=latencies[math.ceil(.95*len(latencies))-1]*1000,
                         raw_request_seconds=[r[1] for r in rows]), indent=2))


if __name__ == "__main__":
    main()
