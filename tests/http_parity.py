#!/usr/bin/env python3
"""Text correctness contract. Uses two running servers; no third-party Python packages."""
import argparse
import base64
import concurrent.futures
import json
import math
import struct
import time
import threading
import urllib.request
import urllib.error


def request(url, texts, dimensions=768, encoding="float"):
    body = json.dumps(dict(model="embeddinggemma-2", input=texts,
                           dimensions=dimensions, encoding_format=encoding)).encode()
    start = time.perf_counter()
    req = urllib.request.Request(url + "/v1/embeddings", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as response:
        obj = json.load(response)
    assert obj["object"] == "list"
    assert obj["usage"]["total_tokens"] == obj["usage"]["prompt_tokens"] > 0
    vectors = []
    for i, item in enumerate(obj["data"]):
        assert item["index"] == i
        v = item["embedding"]
        if encoding == "base64":
            v = struct.unpack("<" + "f" * dimensions, base64.b64decode(v, validate=True))
        assert len(v) == dimensions and all(math.isfinite(x) for x in v)
        assert abs(math.sqrt(sum(x*x for x in v)) - 1) < 1e-4
        vectors.append(v)
    assert len(vectors) == (1 if isinstance(texts, str) else len(texts))
    return vectors, obj["usage"]["total_tokens"], time.perf_counter() - start


def cosine(a, b):
    return sum(x*y for x, y in zip(a, b)) / math.sqrt(sum(x*x for x in a) * sum(x*x for x in b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="http://127.0.0.1:42780")
    ap.add_argument("--cpu", default="http://127.0.0.1:42781")
    ap.add_argument("--long", action="store_true", help="Exercise >1024-token symmetric attention boundary")
    ap.add_argument("--wide", action="store_true", help="Check 16-sequence requests at concurrency 4")
    args = ap.parse_args()
    texts = ["", "hello world", "检索增强生成与向量数据库。", "a " * 37,
             "query: How does a GPU execute a kernel?", "emoji: 🌍 café\nsecond line"]
    if args.long:
        texts = ["boundary " * 1030, "short unrelated sequence"]
    if args.wide:
        texts = [f"wide item {j}: "+"word "*(27+j%7) for j in range(16)]
    gpu, tokens, seconds = request(args.gpu, texts)
    cpu, cpu_tokens, cpu_seconds = request(args.cpu, texts)
    assert tokens == cpu_tokens
    scores = [cosine(a, b) for a, b in zip(gpu, cpu)]
    # CPU uses quantized activations, GPU uses F32 activations. This gate is a
    # regression check, not an independent model-accuracy certification.
    assert min(scores) >= 0.99, scores
    parity = []
    for text, vector in zip(texts, gpu):
        solo, _, _ = request(args.gpu, text)
        parity.append(cosine(vector, solo[0]))
    assert min(parity) >= 0.99999, parity
    # Reordering ragged sequences must not change RoPE positions or attention.
    reversed_vectors, _, _ = request(args.gpu, list(reversed(texts)))
    assert min(cosine(a, b) for a, b in zip(gpu, reversed(reversed_vectors))) >= 0.99999
    for dim in [128, 256, 512, 768]:
        f, _, _ = request(args.gpu, "format test", dim)
        b, _, _ = request(args.gpu, "format test", dim, "base64")
        assert cosine(f[0], b[0]) >= 0.999999
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        concurrent_vectors = list(pool.map(lambda s: request(args.gpu, s)[0][0], texts))
    assert min(cosine(a, b) for a, b in zip(gpu, concurrent_vectors)) >= 0.99999
    wide_parity=[]
    if args.wide:
        # Distinct requests avoid the server's in-flight request coalescing,
        # which is separate from the explicitly disabled persistent caches.
        inputs=[[f"concurrent group {i}: "+text for text in texts] for i in range(4)]
        expected=[request(args.gpu,batch)[0] for batch in inputs]
        barrier=threading.Barrier(4)
        def wide_request(batch):
            barrier.wait(timeout=10)
            return request(args.gpu,batch)[0]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            batches=list(pool.map(wide_request,inputs))
        wide_parity=[min(cosine(a,b) for a,b in zip(ref,batch)) for ref,batch in zip(expected,batches)]
        assert min(wide_parity)>=0.99999, wide_parity
    for payload in [dict(input=[], model="embeddinggemma-2"),
                    dict(input="test", dimensions=17),
                    dict(input="word "*2100, model="embeddinggemma-2")]:
        req = urllib.request.Request(args.gpu+"/v1/embeddings", data=json.dumps(payload).encode(),
                                     headers={"Content-Type":"application/json"})
        try:
            urllib.request.urlopen(req, timeout=60)
            raise AssertionError("invalid request accepted")
        except urllib.error.HTTPError as error:
            assert error.code == 400, error.code
            assert "error" in json.load(error)
    request(args.gpu, "still healthy after validation failures")
    print(json.dumps(dict(test="http_parity", long=args.long, wide=args.wide, tokens=tokens,
                         cpu_cosines=scores, single_batch_cosines=parity,
                         concurrent_batch_cosines=wide_parity,
                         gpu_request_seconds=seconds, cpu_request_seconds=cpu_seconds,
                         note="Correctness run, different CPU/GPU hosts; not a speedup benchmark."), indent=2))


if __name__ == "__main__":
    main()
