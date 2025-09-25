FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \\
    PYTHONUNBUFFERED=1 \\
    HF_HUB_ENABLE_HF_TRANSFER=1

SHELL ["/bin/bash", "-lc"]

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        git \
        unzip && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace
WORKDIR /workspace

# Clone LaMa
RUN git clone --depth=1 https://github.com/advimman/lama.git /workspace/lama

# Python deps
COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir -r /workspace/requirements.txt

WORKDIR /workspace/lama

# App entrypoint
COPY start.sh /workspace/start.sh
COPY app.py /workspace/app.py
RUN chmod +x /workspace/start.sh

EXPOSE 7860

CMD ["bash", "/workspace/start.sh"]


