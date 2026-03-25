"""
Benchmark the FastAPI endpoint end-to-end (upload image -> JSON response).

Measures:
  - End-to-end latency (p50, p95, p99)
  - Preprocessing vs inference breakdown (from server response)
  - Concurrent throughput (N simultaneous requests)

Prerequisites:
  1. Server running: uvicorn Serving.serve:app --port 8000
  2. A test image (any CXR JPEG/PNG)

Usage:
    python Serving/benchmark_api.py \
        --image test_cxr.jpg \
        --url http://localhost:8000 \
        --runs 100 \
        --concurrency 10
"""

import argparse
import asyncio
import time
from pathlib import Path

import httpx
import numpy as np


async def single_request(client: httpx.AsyncClient, url: str, img_bytes: bytes):
    """Send one predict request, return (e2e_ms, preprocess_ms, inference_ms, logits)."""
    t0 = time.perf_counter()
    resp = await client.post(
        f"{url}/predict",
        files={"file": ("image.jpg", img_bytes, "image/jpeg")},
    )
    e2e_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    data = resp.json()
    return e2e_ms, data["preprocessing_ms"], data["inference_ms"], data["logits"]


async def run_sequential(url: str, img_bytes: bytes, warmup: int, runs: int):
    """Sequential requests — measures single-request latency."""
    async with httpx.AsyncClient(timeout=30) as client:
        # warm up
        for _ in range(warmup):
            await single_request(client, url, img_bytes)

        results = []
        for _ in range(runs):
            results.append(await single_request(client, url, img_bytes))
    return results


async def run_concurrent(url: str, img_bytes: bytes, total: int, concurrency: int):
    """Concurrent requests — measures throughput under load."""
    sem = asyncio.Semaphore(concurrency)

    async def bounded_request(client):
        async with sem:
            return await single_request(client, url, img_bytes)

    async with httpx.AsyncClient(timeout=60) as client:
        t0 = time.perf_counter()
        tasks = [bounded_request(client) for _ in range(total)]
        results = await asyncio.gather(*tasks)
        wall_s = time.perf_counter() - t0

    return results, wall_s


def pct(arr, p):
    return float(np.percentile(arr, p))


def print_section(title):
    print(f"\n{'=' * 64}")
    print(title)
    print("=" * 64)


def compute_cpu_reference(image_path: str) -> np.ndarray:
    """Reproduce the server's preprocessing locally and run ONNX CPU inference.
    Returns logits as the parity reference."""
    import onnxruntime as ort
    from PIL import Image

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    img = Image.open(image_path).convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = (arr - MEAN) / STD
    tensor = arr[np.newaxis, ...]

    # Use the FP32 model as ground truth
    fp32_path = str(Path(image_path).parent / "models" / "pacx_mae_fp32.onnx")
    if not Path(fp32_path).exists():
        return None

    sess = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    return sess.run(None, {"image": tensor})[0][0]  # [C]


def main(args):
    img_bytes = Path(args.image).read_bytes()
    print(f"Image: {args.image} ({len(img_bytes) / 1024:.0f} KB)")

    # --- CPU-GPU Parity Check ---
    cpu_ref = compute_cpu_reference(args.image)

    print_section("CPU-GPU PARITY CHECK")
    if cpu_ref is not None:
        # Send N requests and check consistency + parity
        async def _parity_check():
            async with httpx.AsyncClient(timeout=30) as c:
                logits_list = []
                for _ in range(5):
                    _, _, _, logits = await single_request(c, args.url, img_bytes)
                    logits_list.append(logits)
                return logits_list

        server_logits = asyncio.run(_parity_check())

        # 1. Determinism: are all server responses identical?
        arr = np.array(server_logits)
        max_var = np.abs(arr - arr[0]).max()
        det_status = "PASS" if max_var < 1e-6 else ("WARN" if max_var < 1e-3 else "FAIL")
        print(f"  Determinism (5 requests):  max_var={max_var:.2e}  {det_status}")

        # 2. Parity: does server match CPU FP32 reference?
        server_avg = arr.mean(axis=0)
        ref = np.array(cpu_ref)
        max_diff = np.abs(server_avg - ref).max()
        par_status = "PASS" if max_diff < 1e-3 else ("WARN" if max_diff < 1e-1 else "FAIL")
        print(f"  CPU-Server parity:         max_diff={max_diff:.2e}  {par_status}")
        print(f"  CPU ref logits:   {ref.tolist()}")
        print(f"  Server logits:    {server_avg.tolist()}")
    else:
        print("  Skipped — Serving/models/pacx_mae_fp32.onnx not found for reference")

    # --- Sequential latency ---
    print_section(f"SEQUENTIAL LATENCY — {args.runs} requests")
    results = asyncio.run(run_sequential(args.url, img_bytes, warmup=10, runs=args.runs))

    e2e = [r[0] for r in results]
    pre = [r[1] for r in results]
    inf = [r[2] for r in results]

    header = f"{'Metric':<20} {'Mean':>8} {'p50':>8} {'p95':>8} {'p99':>8}"
    print(header)
    print("-" * len(header))
    for name, arr in [("End-to-end (ms)", e2e), ("Preprocess (ms)", pre), ("Inference (ms)", inf)]:
        a = np.array(arr)
        print(f"{name:<20} {a.mean():>8.1f} {pct(a, 50):>8.1f} {pct(a, 95):>8.1f} {pct(a, 99):>8.1f}")

    print(f"\n  Sequential throughput: {1000.0 / np.mean(e2e):.1f} req/s")

    # --- Concurrent throughput ---
    if args.concurrency > 1:
        total_concurrent = args.runs
        print_section(f"CONCURRENT THROUGHPUT — {total_concurrent} requests, concurrency={args.concurrency}")

        results_c, wall_s = asyncio.run(
            run_concurrent(args.url, img_bytes, total_concurrent, args.concurrency)
        )

        e2e_c = [r[0] for r in results_c]
        a = np.array(e2e_c)
        print(f"  Wall time:  {wall_s:.1f}s for {total_concurrent} requests")
        print(f"  Throughput: {total_concurrent / wall_s:.1f} req/s")
        print(f"  Latency:    p50={pct(a, 50):.0f}ms  p95={pct(a, 95):.0f}ms  p99={pct(a, 99):.0f}ms")

    # --- Health + Metrics ---
    import httpx as httpx_sync

    with httpx_sync.Client() as c:
        health = c.get(f"{args.url}/health").json()
        metrics = c.get(f"{args.url}/metrics").json()
    print(f"\n  Server health: {health}")
    print(f"  Server metrics: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True,
                        help="Path to a test CXR image (JPEG/PNG)")
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Number of concurrent requests for throughput test")
    args = parser.parse_args()
    main(args)
