# Unitra ML Service - Self-hosted Docker Image
#
# This Dockerfile creates a standalone ML translation service that can be
# deployed to any GPU-enabled server (Coolify, Kubernetes, Docker Compose, etc.)
#
# Requirements:
# - NVIDIA GPU with CUDA support
# - nvidia-docker2 or NVIDIA Container Toolkit
# - At least 16GB GPU VRAM (for MADLAD-400-3B model)
#
# Build:
#   docker build -t unitra-ml:latest .
#
# Run:
#   docker run --gpus all -p 8001:8001 unitra-ml:latest
#
# Environment Variables:
#   - MODEL_CACHE_DIR: Model cache directory (default: /models)
#   - HUGGINGFACE_TOKEN: Optional HuggingFace token for gated models
#   - PORT: Service port (default: 8001)
#   - HOST: Service host (default: 0.0.0.0)
#   - WORKERS: Number of Uvicorn workers (default: 1)
#   - LOG_LEVEL: Logging level (default: info)
#

# Base image with CUDA support
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Create app directory
WORKDIR /app

# Create non-root user
RUN useradd -m -s /bin/bash appuser

# Install Python dependencies
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt

# Copy application code
COPY src/ ./src/
COPY selfhosted/ ./selfhosted/

# Create model cache directory
RUN mkdir -p /models && chown -R appuser:appuser /models /app

# Switch to non-root user
USER appuser

# Environment defaults
ENV MODEL_CACHE_DIR=/models
ENV PORT=8001
ENV HOST=0.0.0.0
ENV WORKERS=1
ENV LOG_LEVEL=info
ENV HF_HOME=/models/huggingface

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start service
CMD ["python", "-m", "selfhosted.main"]
