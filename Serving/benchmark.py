"""
Benchmark PyTorch vs ONNX Runtime inference for PaCX-MAE.

Tests all exported variants (FP32, FP16, INT8) on CPU and GPU,
including IO Binding for GPU and batched throughput.

Usage:
    LD_LIBRARY_PATH=<nvidia_libs> python Serving/benchmark.py \
        --mode pacx \
        --ckpt_path checkpoints/Final_PACX_MAE.ckpt \
        --mae_path checkpoints/mae/mae_cxr_final.ckpt \
        --model_dir Serving/models \
        --num_classes 1 \
        --runs 200
"""

import argparse
import os
import sys
import time
import resource
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Models.models import CXRModel


def get_peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def latency_stats(times_ms):
    arr = np.array(times_ms)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def bench_pytorch(model, dummy, warmup, runs, device):
    model = model.to(device)
    inp = dummy.to(device)
    model.eval()

    with torch.no_grad():
        for _ in range(warmup):
            model(inp)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(inp)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    with torch.no_grad():
        ref_output = model(inp).detach().cpu().numpy()
    return times, ref_output


def bench_onnx(onnx_path, dummy_np, warmup, runs, provider):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    available = ort.get_available_providers()
    if provider not in available:
        return None, None

    try:
        sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=[provider])
        # Check if actually loaded on requested provider
        if provider == "CUDAExecutionProvider" and sess.get_providers()[0] != "CUDAExecutionProvider":
            return None, None
    except Exception:
        return None, None

    for _ in range(warmup):
        sess.run(None, {"image": dummy_np})

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {"image": dummy_np})
        times.append((time.perf_counter() - t0) * 1000)

    ref_output = sess.run(None, {"image": dummy_np})[0]
    return times, ref_output


def bench_onnx_iobinding(onnx_path, dummy_np, warmup, runs, num_classes):
    """GPU benchmark using IO Binding — avoids CPU<->GPU copies per request."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    provider = "CUDAExecutionProvider"
    available = ort.get_available_providers()
    if provider not in available:
        return None, None

    try:
        sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=[provider])
        if sess.get_providers()[0] != "CUDAExecutionProvider":
            return None, None
    except Exception:
        return None, None

    device_id = 0
    batch_size = dummy_np.shape[0]

    # Pre-allocate GPU tensors
    input_ort = ort.OrtValue.ortvalue_from_numpy(dummy_np, "cuda", device_id)
    output_ort = ort.OrtValue.ortvalue_from_shape_and_type(
        [batch_size, num_classes], np.float32, "cuda", device_id
    )

    binding = sess.io_binding()
    binding.bind_ortvalue_input("image", input_ort)
    binding.bind_ortvalue_output("logits", output_ort)

    # Warm up
    for _ in range(warmup):
        sess.run_with_iobinding(binding)

    times = []
    for _ in range(runs):
        # Re-bind input each iteration (simulates new data arriving on GPU)
        input_ort = ort.OrtValue.ortvalue_from_numpy(dummy_np, "cuda", device_id)
        binding.bind_ortvalue_input("image", input_ort)
        t0 = time.perf_counter()
        sess.run_with_iobinding(binding)
        times.append((time.perf_counter() - t0) * 1000)

    ref_output = output_ort.numpy()
    return times, ref_output


def bench_onnx_batched(onnx_path, batch_sizes, warmup, runs, provider, num_classes,
                       use_iobinding=False):
    """Measure throughput at different batch sizes. Returns {bs: img_per_sec}."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    available = ort.get_available_providers()
    if provider not in available:
        return {}

    try:
        sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=[provider])
        if provider == "CUDAExecutionProvider" and sess.get_providers()[0] != "CUDAExecutionProvider":
            return {}
    except Exception:
        return {}

    results = {}
    for bs in batch_sizes:
        dummy = np.random.randn(bs, 3, 224, 224).astype(np.float32)

        if use_iobinding and provider == "CUDAExecutionProvider":
            input_ort = ort.OrtValue.ortvalue_from_numpy(dummy, "cuda", 0)
            output_ort = ort.OrtValue.ortvalue_from_shape_and_type(
                [bs, num_classes], np.float32, "cuda", 0
            )
            binding = sess.io_binding()
            binding.bind_ortvalue_input("image", input_ort)
            binding.bind_ortvalue_output("logits", output_ort)

            for _ in range(warmup):
                sess.run_with_iobinding(binding)

            times = []
            for _ in range(runs):
                input_ort = ort.OrtValue.ortvalue_from_numpy(dummy, "cuda", 0)
                binding.bind_ortvalue_input("image", input_ort)
                t0 = time.perf_counter()
                sess.run_with_iobinding(binding)
                times.append((time.perf_counter() - t0) * 1000)
        else:
            for _ in range(warmup):
                sess.run(None, {"image": dummy})
            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                sess.run(None, {"image": dummy})
                times.append((time.perf_counter() - t0) * 1000)

        mean_ms = np.mean(times)
        results[bs] = bs * 1000.0 / mean_ms  # img/s

    return results


def print_table(rows, col_widths):
    for row in rows:
        line = ""
        for i, val in enumerate(row):
            line += str(val).ljust(col_widths[i])
        print(line)


def main(args):
    warmup = 20
    runs = args.runs
    dummy = torch.randn(1, 3, 224, 224)
    dummy_np = dummy.numpy()

    # --- Build PyTorch model ---
    print("Loading PyTorch model ...")
    model = CXRModel(
        num_classes=args.num_classes,
        mode=args.mode,
        model_checkpoints=args.ckpt_path,
        mae_path=args.mae_path,
        unfreeze_backbone=False,
        task="sl",
    )
    model.eval()

    # --- Re-export ONNX variants from the same model (identical head weights) ---
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    fp32_path = str(model_dir / "_bench_fp32.onnx")
    print("Re-exporting ONNX from benchmark model for fair comparison ...")
    torch.onnx.export(
        model, dummy, fp32_path, opset_version=17,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )

    # FP16
    fp16_path = str(model_dir / "_bench_fp16.onnx")
    import onnx
    from onnxruntime.transformers import float16
    model_fp32 = onnx.load(fp32_path)
    model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=True)
    onnx.save(model_fp16, fp16_path)

    # INT8
    int8_path = str(model_dir / "_bench_int8.onnx")
    from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process
    preprocessed_path = str(model_dir / "_bench_preprocessed.onnx")
    quant_pre_process(fp32_path, preprocessed_path)
    quantize_dynamic(preprocessed_path, int8_path, weight_type=QuantType.QUInt8)
    Path(preprocessed_path).unlink(missing_ok=True)

    print(f"  FP32: {Path(fp32_path).stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  FP16: {Path(fp16_path).stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  INT8: {Path(int8_path).stat().st_size / 1024 / 1024:.1f} MB")

    results = {}

    # --- PyTorch CPU ---
    print("\nBenchmarking: PyTorch CPU ...")
    t, o = bench_pytorch(model, dummy, warmup, runs, torch.device("cpu"))
    results["PyTorch CPU"] = {"stats": latency_stats(t), "output": o}

    # --- PyTorch GPU ---
    if torch.cuda.is_available():
        print("Benchmarking: PyTorch GPU ...")
        t, o = bench_pytorch(model, dummy, warmup, runs, torch.device("cuda"))
        results["PyTorch GPU"] = {"stats": latency_stats(t), "output": o}

    # --- ONNX variants (standard) ---
    onnx_configs = [
        ("ONNX FP32 CPU",  fp32_path, "CPUExecutionProvider"),
        ("ONNX FP32 GPU",  fp32_path, "CUDAExecutionProvider"),
        ("ONNX FP16 GPU",  fp16_path, "CUDAExecutionProvider"),
        ("ONNX INT8 CPU",  int8_path, "CPUExecutionProvider"),
    ]

    for name, path, provider in onnx_configs:
        print(f"Benchmarking: {name} ...")
        t, o = bench_onnx(path, dummy_np, warmup, runs, provider)
        if t is None:
            print(f"  Skipped ({provider} not available)")
            continue
        results[name] = {"stats": latency_stats(t), "output": o}

    # --- IO Binding GPU benchmarks ---
    print("Benchmarking: ONNX FP32 GPU IOBind ...")
    t, o = bench_onnx_iobinding(fp32_path, dummy_np, warmup, runs, args.num_classes)
    if t is not None:
        results["ONNX FP32 GPU IOBind"] = {"stats": latency_stats(t), "output": o}
    else:
        print("  Skipped (CUDA not available)")

    print("Benchmarking: ONNX FP16 GPU IOBind ...")
    t, o = bench_onnx_iobinding(fp16_path, dummy_np, warmup, runs, args.num_classes)
    if t is not None:
        results["ONNX FP16 GPU IOBind"] = {"stats": latency_stats(t), "output": o}
    else:
        print("  Skipped (CUDA not available)")

    # ========== REPORT: CORRECTNESS ==========
    ref = results["PyTorch CPU"]["output"]

    print("\n" + "=" * 80)
    print("CORRECTNESS — max |diff| vs PyTorch CPU reference")
    print("=" * 80)
    rows = [["Backend", "Max Abs Diff", "Status"]]
    for name, r in results.items():
        diff = float(np.abs(ref - r["output"]).max())
        status = "PASS" if diff < 1e-4 else ("WARN" if diff < 1e-2 else "FAIL")
        rows.append([name, f"{diff:.2e}", status])
    print_table(rows, [28, 16, 8])

    # ========== REPORT: LATENCY ==========
    print("\n" + "=" * 80)
    print(f"LATENCY (ms) — {runs} runs, batch=1, image=224x224")
    print("=" * 80)
    rows = [["Backend", "Mean", "p50", "p95", "p99", "img/s"]]
    for name, r in results.items():
        s = r["stats"]
        ips = 1000.0 / s["mean"] if s["mean"] > 0 else 0
        rows.append([
            name,
            f"{s['mean']:.1f}",
            f"{s['p50']:.1f}",
            f"{s['p95']:.1f}",
            f"{s['p99']:.1f}",
            f"{ips:.1f}",
        ])
    print_table(rows, [28, 10, 10, 10, 10, 10])

    # Speedup summary
    pt_cpu = results["PyTorch CPU"]["stats"]["mean"]
    print(f"\n  Speedup vs PyTorch CPU ({pt_cpu:.1f}ms):")
    for name, r in results.items():
        if name == "PyTorch CPU":
            continue
        s = r["stats"]["mean"]
        print(f"    {name:<26} {pt_cpu / s:>6.2f}x  ({s:.1f}ms)")

    # ========== REPORT: BATCHED THROUGHPUT ==========
    batch_sizes = [1, 2, 4, 8, 16]
    batch_runs = 50

    print("\n" + "=" * 80)
    print(f"BATCHED THROUGHPUT (img/s) — {batch_runs} runs per batch size")
    print("=" * 80)

    batch_configs = [
        ("ONNX FP32 CPU",          fp32_path, "CPUExecutionProvider",  False),
        ("ONNX INT8 CPU",          int8_path, "CPUExecutionProvider",  False),
        ("ONNX FP32 GPU",          fp32_path, "CUDAExecutionProvider", False),
        ("ONNX FP32 GPU IOBind",   fp32_path, "CUDAExecutionProvider", True),
        ("ONNX FP16 GPU IOBind",   fp16_path, "CUDAExecutionProvider", True),
    ]

    all_batch_results = {}
    for name, path, provider, iob in batch_configs:
        print(f"  Batched: {name} ...")
        br = bench_onnx_batched(path, batch_sizes, warmup, batch_runs, provider,
                                args.num_classes, use_iobinding=iob)
        if br:
            all_batch_results[name] = br

    if all_batch_results:
        header = ["Backend"] + [f"bs={bs}" for bs in batch_sizes]
        widths = [28] + [12] * len(batch_sizes)
        rows = [header]
        for name, br in all_batch_results.items():
            row = [name] + [f"{br.get(bs, 0):.0f}" for bs in batch_sizes]
            rows.append(row)
        print_table(rows, widths)

    # Clean up temp benchmark models
    for p in [fp32_path, fp16_path, int8_path]:
        try:
            os.unlink(p)
        except OSError:
            pass

    # Model sizes
    real_dir = Path(args.model_dir)
    print(f"\n  Model sizes:")
    for label, fname in [("FP32", "pacx_mae_fp32.onnx"), ("FP16", "pacx_mae_fp16.onnx"), ("INT8", "pacx_mae_int8.onnx")]:
        p = real_dir / fname
        if p.exists():
            print(f"    {label}: {p.stat().st_size / 1024 / 1024:.1f} MB")

    print(f"  Peak RSS: {get_peak_rss_mb():.0f} MB\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs ONNX (all variants)")
    parser.add_argument("--mode", type=str, default="pacx")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--mae_path", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default="Serving/models")
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()
    main(args)
