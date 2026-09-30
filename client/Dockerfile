# CUDA-native base: provides libcudart.so, libcublas.so, cuDNN required by onnxruntime-gpu.
# python:3.10-slim lacks these C++ binaries — CUDAExecutionProvider silently falls back to CPU.
# Ubuntu 22.04 ships Python 3.10 natively; no manual version pinning needed.
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Prevent interactive timezone/locale prompts during apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system deps BEFORE copying app code to maximize layer cache reuse.
# curl retained for health checks; pip upgraded to avoid metadata format warnings.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --upgrade pip

# Symlink python → python3 for scripts that invoke bare `python`
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install Python deps in a separate layer — cached unless requirements.txt changes
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application payload last (most frequently changing layer)
COPY . .