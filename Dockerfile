#Bismillah
# Qwen TTS Worker for RunPod Serverless
# Uses Flash Attention 2 and bfloat16 optimizations

# Use RunPod's pre-built PyTorch image (has CUDA, Python 3.11, PyTorch ready)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Qwen model weights during build to optimize cold starts
# We use the huggingface-cli instead of Python script to avoid Flash-Attn CPU compilation errors
RUN huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice

# Copy handler
COPY handler.py /app/handler.py

# Start handler
CMD ["python", "-u", "/app/handler.py"]
