FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY Serving /app/Serving

EXPOSE 8000

ENV ONNX_MODEL_PATH=/app/Serving/models/pacx_mae_int8.onnx
ENV ONNX_PROVIDER=CPUExecutionProvider
ENV BATCH_MAX_SIZE=1
ENV BATCH_MAX_WAIT_MS=1
ENV ORT_NUM_THREADS=1

CMD ["uvicorn", "Serving.serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]