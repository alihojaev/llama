FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=1

SHELL ["/bin/bash", "-lc"]

# System deps and Python 3
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        git \
        unzip \
        python3 \
        python3-venv \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace
WORKDIR /workspace

# Clone LaMa
RUN git clone --depth=1 https://github.com/advimman/lama.git /workspace/lama

# Python deps
COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir -r /workspace/requirements.txt && \
    python3 -m pip install --no-cache-dir torch==1.8.0 torchvision==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /workspace/lama

# App entrypoint
COPY start.sh /workspace/start.sh
COPY app.py /workspace/app.py
COPY handler.py /workspace/handler.py
RUN chmod +x /workspace/start.sh

EXPOSE 7860

CMD ["bash", "/workspace/start.sh"]


