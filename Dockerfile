FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-lc"]

# System deps and Python 3.8
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        ca-certificates \
        git \
        unzip && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.8 \
        python3.8-venv \
        python3.8-distutils && \
    rm -rf /var/lib/apt/lists/*

# Pip for Python 3.8
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    python3.8 /tmp/get-pip.py && \
    rm -f /tmp/get-pip.py

RUN mkdir -p /workspace
WORKDIR /workspace

# Clone LaMa
RUN git clone --depth=1 https://github.com/advimman/lama.git /workspace/lama

# Python deps
COPY requirements.txt /workspace/requirements.txt
RUN python3.8 -m pip install --no-cache-dir -r /workspace/requirements.txt && \
    python3.8 -m pip install --no-cache-dir torch==1.8.0 torchvision==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /workspace/lama

# App entrypoint
COPY start.sh /workspace/start.sh
COPY app.py /workspace/app.py
RUN chmod +x /workspace/start.sh

EXPOSE 7860

CMD ["bash", "/workspace/start.sh"]


