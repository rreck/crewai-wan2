# Copyright (c) RRECKTEK LLC
# Version: 1.0.0
# Built: @EPOCH

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ca-certificates \
    wget \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch with CUDA support (torch >= 2.4.0 required for Wan2)
RUN pip3 install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Install Wan2 via diffusers and dependencies
RUN pip3 install --no-cache-dir \
    diffusers>=0.30.0 \
    transformers>=4.40.0 \
    accelerate>=0.29.0 \
    huggingface_hub[cli]

# Install flash-attention (required for Wan2)
RUN pip3 install --no-cache-dir flash-attn --no-build-isolation

# Create application directory and copy main script
WORKDIR /opt/app
COPY app/main.py /opt/app/

# Create necessary directories including kb for pmem 1.0
RUN mkdir -p /work/input /work/output /work/output/logs /work/kb/short /work/kb/long /var/run /var/log

# Expose metrics and API ports
EXPOSE 9093 8083

# Default environment variables
ENV INPUT_DIR=/work/input \
    OUTPUT_DIR=/work/output \
    WAN2_MODEL=Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    PIDFILE=/var/run/wan2-agent.pid \
    METRICS_PORT=9093 \
    API_PORT=8083 \
    HF_HOME=/work/.cache/huggingface

# Healthcheck: verifies services are running
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:9093/metrics > /dev/null && \
        curl -f http://localhost:8083/health > /dev/null || exit 1

# Default command (can be overridden)
CMD ["python3", "/opt/app/main.py", "--daemon", "-i", "/work/input", "-o", "/work/output"]
