FROM spxiong/pytorch:2.7.1-py3.10.15-cuda12.8.1-ubuntu22.04 AS base

WORKDIR /app

# Set CUDA environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_MODULE_LOADING=LAZY
ENV PYTHONPATH="/app"
ENV TORCH_HOME="/app/checkpoints"
ENV HF_HOME="/app/checkpoints"

# Install system dependencies in single layer (chỉ cần FFmpeg cho core functionality)
RUN apt-get update && apt-get install -y \
    python3.10-dev \
    python3.10-distutils \
    build-essential \
    ffmpeg \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements và install dependencies trong single layer để optimize caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Additional dependencies có thể cần cho runtime
RUN pip install --no-cache-dir \
    imageio-ffmpeg==0.5.1

# Copy source code TRƯỚC khi download models để tận dụng layer caching
COPY . /app/

# Download chỉ LatentSync v1.0 models (loại bỏ face enhancement models)
RUN echo "=== Downloading LatentSync v1.0 core models ===" && \
    mkdir -p /app/checkpoints/whisper && \
    \
    echo "📥 Downloading LatentSync v1.0 models..." && \
    huggingface-cli download ByteDance/LatentSync --local-dir checkpoints && \
    \
    echo "✅ LatentSync v1.0 core models downloaded"

# Verify core model files và cleanup trong một layer
RUN echo "=== Verifying LatentSync v1.0 core files ===" && \
    test -f /app/configs/unet/second_stage.yaml && echo "✅ LatentSync v1.0 config verified" && \
    test -f /app/checkpoints/latentsync_unet.pt && echo "✅ LatentSync v1.0 UNet model verified" && \
    (test -f /app/checkpoints/whisper/tiny.pt && echo "✅ Whisper tiny model verified" || echo "ℹ️ Whisper tiny model not found") && \
    (test -f /app/checkpoints/whisper/small.pt && echo "✅ Whisper small model verified" || echo "ℹ️ Whisper small model not found") && \
    \
    echo "=== Cleanup ===" && \
    rm -rf /tmp/* /var/tmp/* && \
    find /usr/local -name "*.pyc" -delete && \
    find /usr/local -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Health check optimized cho serverless (nhanh hơn)
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=2 \
  CMD python -c "import torch; print('GPU:', torch.cuda.is_available()); assert torch.cuda.is_available()" || exit 1

# Expose port cho RunPod
EXPOSE 8000

CMD ["python", "rp_handler.py"]
