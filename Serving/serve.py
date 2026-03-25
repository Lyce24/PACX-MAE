"""
FastAPI server for PaCX-MAE inference via ONNX Runtime.

Features:
  - Async request batching: accumulates concurrent requests and runs them
    as a single batched inference call for higher GPU utilization.
  - IO Binding on GPU: pre-allocates GPU tensors to avoid CPU<->GPU memcpy.
  - Thread-pool offloaded inference to keep the event loop responsive.

Usage:
    # CPU INT8 (cost-efficient)
    ONNX_MODEL_PATH=Serving/models/pacx_mae_int8.onnx \
    uvicorn Serving.serve:app --host 0.0.0.0 --port 8000

    # GPU FP32 with batching + IO Binding
    ONNX_MODEL_PATH=Serving/models/pacx_mae_fp32.onnx \
    ONNX_PROVIDER=CUDAExecutionProvider \
    uvicorn Serving.serve:app --host 0.0.0.0 --port 8000

Env vars:
    ONNX_MODEL_PATH     Path to .onnx file
    ONNX_PROVIDER        CPUExecutionProvider | CUDAExecutionProvider
    BATCH_MAX_SIZE       Max batch size before forced flush (default: 8)
    BATCH_MAX_WAIT_MS    Max wait time to accumulate a batch (default: 10)
    ORT_NUM_THREADS      Intra-op threads for CPU (default: 0 = ORT default)
"""

import asyncio
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

# --- Config ---
ONNX_PATH = os.getenv("ONNX_MODEL_PATH", "Serving/models/pacx_mae_fp32.onnx")
PROVIDER = os.getenv("ONNX_PROVIDER", "CPUExecutionProvider")
NUM_THREADS = int(os.getenv("ORT_NUM_THREADS", "0"))
BATCH_MAX_SIZE = int(os.getenv("BATCH_MAX_SIZE", "8"))
BATCH_MAX_WAIT_MS = float(os.getenv("BATCH_MAX_WAIT_MS", "10"))

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


# ---------- Batch Scheduler ----------

@dataclass
class _PendingRequest:
    tensor: np.ndarray  # [1, 3, 224, 224]
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())


class BatchScheduler:
    """Accumulates incoming requests and flushes them as a single batch."""

    def __init__(self, session: ort.InferenceSession, max_size: int,
                 max_wait_ms: float, executor: ThreadPoolExecutor,
                 use_iobinding: bool, num_classes: int):
        self.session = session
        self.max_size = max_size
        self.max_wait_s = max_wait_ms / 1000.0
        self.executor = executor
        self.use_iobinding = use_iobinding
        self.num_classes = num_classes
        self._queue: list[_PendingRequest] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None

    async def submit(self, tensor: np.ndarray) -> np.ndarray:
        """Submit a single [1,3,224,224] tensor. Returns [1, C] logits."""
        req = _PendingRequest(tensor=tensor)
        async with self._lock:
            self._queue.append(req)
            if len(self._queue) >= self.max_size:
                await self._flush()
            elif self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(self._delayed_flush())
        return await req.future

    async def _delayed_flush(self):
        """Wait up to max_wait then flush whatever we have."""
        await asyncio.sleep(self.max_wait_s)
        async with self._lock:
            if self._queue:
                await self._flush()

    async def _flush(self):
        """Run batched inference on all queued requests."""
        if not self._queue:
            return
        batch = self._queue
        self._queue = []
        self._flush_task = None

        tensors = np.concatenate([r.tensor for r in batch], axis=0)  # [B,3,224,224]
        loop = asyncio.get_event_loop()

        if self.use_iobinding:
            logits = await loop.run_in_executor(
                self.executor, self._run_iobinding, tensors
            )
        else:
            logits = await loop.run_in_executor(
                self.executor, self._run_standard, tensors
            )

        # Dispatch results back to individual futures
        for i, req in enumerate(batch):
            req.future.set_result(logits[i:i+1])

    def _run_standard(self, tensors: np.ndarray) -> np.ndarray:
        return self.session.run(None, {"image": tensors})[0]

    def _run_iobinding(self, tensors: np.ndarray) -> np.ndarray:
        bs = tensors.shape[0]
        input_ort = ort.OrtValue.ortvalue_from_numpy(tensors, "cuda", 0)
        output_ort = ort.OrtValue.ortvalue_from_shape_and_type(
            [bs, self.num_classes], np.float32, "cuda", 0
        )
        binding = self.session.io_binding()
        binding.bind_ortvalue_input("image", input_ort)
        binding.bind_ortvalue_output("logits", output_ort)
        self.session.run_with_iobinding(binding)
        return output_ort.numpy()


# ---------- App State ----------

state: dict = {
    "session": None,
    "scheduler": None,
    "executor": None,
    "request_count": 0,
    "total_inference_ms": 0.0,
    "batches_run": 0,
    "total_batch_size": 0,
}


def _build_session() -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if NUM_THREADS > 0:
        opts.intra_op_num_threads = NUM_THREADS

    providers = [PROVIDER]
    available = ort.get_available_providers()
    if PROVIDER not in available:
        print(f"  WARNING: {PROVIDER} not available, falling back to CPU. "
              f"Available: {available}")
        providers = ["CPUExecutionProvider"]

    return ort.InferenceSession(ONNX_PATH, sess_options=opts, providers=providers)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading ONNX model from {ONNX_PATH} ...")
    session = _build_session()
    state["session"] = session
    state["executor"] = ThreadPoolExecutor(max_workers=4)

    active_provider = session.get_providers()[0]
    use_iobinding = (active_provider == "CUDAExecutionProvider")

    # Determine num_classes from model output shape
    output_shape = session.get_outputs()[0].shape
    num_classes = output_shape[1] if len(output_shape) > 1 else 1

    state["scheduler"] = BatchScheduler(
        session=session,
        max_size=BATCH_MAX_SIZE,
        max_wait_ms=BATCH_MAX_WAIT_MS,
        executor=state["executor"],
        use_iobinding=use_iobinding,
        num_classes=num_classes,
    )

    # Warm up
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    session.run(None, {"image": dummy})

    mode = "IO Binding + Batching" if use_iobinding else "Batching"
    print(f"Model loaded. Provider: {active_provider} | Mode: {mode} "
          f"| max_batch={BATCH_MAX_SIZE} max_wait={BATCH_MAX_WAIT_MS}ms")
    yield
    state["executor"].shutdown(wait=False)
    state["session"] = None


app = FastAPI(title="PaCX-MAE Inference API", lifespan=lifespan)


# ---------- Helpers ----------

def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = (arr - MEAN) / STD
    return arr[np.newaxis, ...]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------- Endpoints ----------

class PredictResponse(BaseModel):
    logits: list[float]
    probabilities: list[float]
    preprocessing_ms: float
    inference_ms: float


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if state["session"] is None:
        raise HTTPException(503, "Model not loaded")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(400, "Could not decode image")

    t0 = time.perf_counter()
    tensor = preprocess(image)
    preprocess_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    logits = await state["scheduler"].submit(tensor)
    inference_ms = (time.perf_counter() - t1) * 1000

    state["request_count"] += 1
    state["total_inference_ms"] += inference_ms

    logits_flat = logits[0].tolist()
    probs = sigmoid(logits[0]).tolist()

    return PredictResponse(
        logits=logits_flat,
        probabilities=probs,
        preprocessing_ms=round(preprocess_ms, 2),
        inference_ms=round(inference_ms, 2),
    )


@app.get("/health")
async def health():
    session = state["session"]
    provider = session.get_providers()[0] if session else "N/A"
    scheduler = state["scheduler"]
    return {
        "status": "ok",
        "model_loaded": session is not None,
        "provider": provider,
        "batching": scheduler is not None,
        "io_binding": scheduler.use_iobinding if scheduler else False,
    }


class MetricsResponse(BaseModel):
    onnx_path: str
    provider: str
    total_requests: int
    avg_inference_ms: float


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    count = state["request_count"]
    avg = state["total_inference_ms"] / count if count > 0 else 0.0
    session = state["session"]
    provider = session.get_providers()[0] if session else PROVIDER
    return MetricsResponse(
        onnx_path=ONNX_PATH,
        provider=provider,
        total_requests=count,
        avg_inference_ms=round(avg, 2),
    )
