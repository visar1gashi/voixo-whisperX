# WhisperX Serverless (RunPod) — CUDA 11.8, Torch 2.1.0, no VAD
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git curl ca-certificates libsndfile1 python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Python
RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Pin GPU Torch to CUDA 11.8 wheels to avoid torch 2.6+ safe-load issues
RUN pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.1.0+cu118 torchaudio==2.1.0+cu118 torchvision==0.16.0+cu118

# WhisperX from GitHub (brings dependencies)
RUN pip install git+https://github.com/m-bain/whisperx.git

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
