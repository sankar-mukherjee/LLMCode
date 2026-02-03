FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV PYTHONPATH=/workspace
WORKDIR /workspace

ARG DEBIAN_FRONTEND=noninteractive

# Set Hugging Face cache directories to persistent local folders
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
ENV HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub

# Install Python and essential tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

# Install PyTorch with CUDA support (smaller than full NVIDIA image)
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

COPY . .
