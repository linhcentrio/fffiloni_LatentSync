FROM spxiong/pytorch:2.7.1-py3.10.15-cuda12.8.1-ubuntu22.04 AS base

WORKDIR /app

# Set CUDA environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_MODULE_LOADING=LAZY

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10-dev \
    python3.10-distutils \
    build-essential \
    libgl1 \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /app/

# Upgrade pip and install tools
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel

# Install basic dependencies first
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
    scikit-image \
    Pillow \
    matplotlib \
    scipy \
    easydict \
    cython \
    protobuf \
    onnx

# Install InsightFace
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "=== Installing InsightFace ===" && \
    pip install --no-cache-dir insightface==0.7.3 || \
    (echo "=== Pip install failed, downloading wheel ===" && \
    wget --no-check-certificate --timeout=30 --tries=3 \
    "https://huggingface.co/deauxpas/colabrepo/resolve/main/insightface-0.7.3-cp310-cp310-linux_x86_64.whl" \
    -O /tmp/insightface-0.7.3-cp310-cp310-linux_x86_64.whl && \
    pip install /tmp/insightface-0.7.3-cp310-cp310-linux_x86_64.whl && \
    rm -f /tmp/insightface-0.7.3-cp310-cp310-linux_x86_64.whl)

# Install main dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
    diffusers==0.32.2 \
    transformers==4.48.0 \
    decord==0.6.0 \
    accelerate==0.26.1 \
    einops==0.7.0 \
    omegaconf==2.3.0 \
    opencv-python==4.9.0.80 \
    mediapipe==0.10.11 \
    python_speech_features==0.6 \
    librosa==0.10.1 \
    scenedetect==0.6.1 \
    ffmpeg-python==0.2.0 \
    imageio==2.31.1 \
    imageio-ffmpeg==0.5.1 \
    lpips==0.1.4 \
    face-alignment==1.4.1 \
    huggingface-hub==0.30.2 \
    kornia==0.8.0 \
    onnxruntime-gpu==1.21.0 \
    runpod>=1.6.0 \
    minio>=7.0.0 \
    moviepy>=1.0.3 \
    pydub>=0.25.1

# Upgrade numpy last if needed
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade numpy==1.24.3

# Copy source code
COPY . /app/

# Download LatentSync v1.0 models
RUN echo "=== Downloading LatentSync v1.0 models ===" && \
    mkdir -p /app/checkpoints/whisper && \
    echo "📥 Downloading LatentSync v1.0 models..." && \
    huggingface-cli download ByteDance/LatentSync --local-dir checkpoints && \
    echo "✅ LatentSync v1.0 models downloaded"

# Verify all model files exist
RUN echo "=== Verifying model files ===" && \
    test -f /app/configs/unet/second_stage.yaml && echo "✅ LatentSync v1.0 config verified" && \
    test -f /app/checkpoints/latentsync_unet.pt && echo "✅ LatentSync v1.0 UNet model verified" && \
    (test -f /app/checkpoints/whisper/tiny.pt && echo "✅ Whisper tiny model verified" || echo "ℹ️ Whisper tiny model not found") && \
    (test -f /app/checkpoints/whisper/small.pt && echo "✅ Whisper small model verified" || echo "ℹ️ Whisper small model not found")

# Set environment variables
ENV PYTHONPATH="/app"
ENV TORCH_HOME="/app/checkpoints"
ENV HF_HOME="/app/checkpoints"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

CMD ["python", "rp_handler.py"]
