# CUDA 12.6 + cuDNN 9 + PyTorch 2.6 runtime
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/huggingface \
    HUGGINGFACE_HUB_CACHE=/data/huggingface \
    TRANSFORMERS_CACHE=/data/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

WORKDIR /app

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends git wget && \
    rm -rf /var/lib/apt/lists/*

# Python deps
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir \
        "transformers>=4.51.0" \
        "accelerate>=0.33.0" \
        "sentencepiece>=0.2.0" \
        "huggingface_hub[hf_transfer]>=0.25.0" \
        "hf_transfer>=0.1.6" \
        "runpod>=1.7.0"

# Worker code
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
