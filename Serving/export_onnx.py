"""
Export PaCX-MAE CXRModel to ONNX (FP32), then optionally produce FP16 and INT8 variants.

Usage:
    python Serving/export_onnx.py \
        --mode pacx \
        --ckpt_path checkpoints/Final_PACX_MAE.ckpt \
        --mae_path checkpoints/mae/mae_cxr_final.ckpt \
        --num_classes 1 \
        --output_dir Serving/models
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Models.models import CXRModel


def verify(onnx_path, dummy_np, pt_out, label=""):
    """Run ONNX model and compare against PyTorch reference."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"image": dummy_np})[0]
    max_diff = float(np.abs(pt_out - onnx_out).max())
    size_mb = Path(onnx_path).stat().st_size / 1024 / 1024
    status = "PASS" if max_diff < 1e-3 else "WARN"
    print(f"  [{label:>5}]  max_diff={max_diff:.2e}  {status}  size={size_mb:.1f} MB  -> {onnx_path}")
    return max_diff


def export(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Build model ---
    print(f"[1/5] Building CXRModel (mode={args.mode}) ...")
    model = CXRModel(
        num_classes=args.num_classes,
        mode=args.mode,
        model_checkpoints=args.ckpt_path,
        mae_path=args.mae_path,
        unfreeze_backbone=False,
        task="sl",
    )
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        pt_out = model(dummy).detach().numpy()
    dummy_np = dummy.numpy()

    # --- 2. Export FP32 ONNX ---
    fp32_path = str(out_dir / "pacx_mae_fp32.onnx")
    print(f"[2/5] Exporting FP32 ONNX ...")
    torch.onnx.export(
        model, dummy, fp32_path, opset_version=17,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )
    verify(fp32_path, dummy_np, pt_out, "FP32")

    # --- 3. FP16 conversion ---
    fp16_path = str(out_dir / "pacx_mae_fp16.onnx")
    print(f"[3/5] Converting to FP16 ...")
    from onnxruntime.transformers import float16

    model_fp32 = onnx.load(fp32_path)
    model_fp16 = float16.convert_float_to_float16(
        model_fp32,
        keep_io_types=True,  # keep input/output as FP32 for easy interop
    )
    onnx.save(model_fp16, fp16_path)
    verify(fp16_path, dummy_np, pt_out, "FP16")

    # --- 4. INT8 dynamic quantization ---
    int8_path = str(out_dir / "pacx_mae_int8.onnx")
    print(f"[4/5] Quantizing to INT8 (dynamic) ...")
    from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process

    # Pre-process (shape inference + optimization) required for INT8 with GPU runtime
    preprocessed_path = str(out_dir / "_preprocessed.onnx")
    quant_pre_process(fp32_path, preprocessed_path)

    quantize_dynamic(
        preprocessed_path,
        int8_path,
        weight_type=QuantType.QUInt8,  # QUInt8 has broader op coverage than QInt8
    )
    # clean up intermediate
    Path(preprocessed_path).unlink(missing_ok=True)
    verify(int8_path, dummy_np, pt_out, "INT8")

    # --- 5. Summary ---
    print(f"\n[5/5] Export summary:")
    print(f"  {'Variant':<8} {'Size (MB)':>10} {'Path'}")
    print(f"  {'-'*8} {'-'*10} {'-'*40}")
    for label, path in [("FP32", fp32_path), ("FP16", fp16_path), ("INT8", int8_path)]:
        sz = Path(path).stat().st_size / 1024 / 1024
        print(f"  {label:<8} {sz:>10.1f} {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CXRModel to ONNX (FP32 + FP16 + INT8)")
    parser.add_argument("--mode", type=str, default="pacx",
                        choices=["imagenet", "mae", "pacx"])
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--mae_path", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="Serving/models")
    args = parser.parse_args()
    export(args)
