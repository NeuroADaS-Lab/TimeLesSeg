FROM ubuntu:22.04

RUN apt-get -y update && \
    apt-get install -y --no-install-recommends \
    git \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python-is-python3 && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth=1 https://github.com/NeuroADaS-Lab/TimeLesSeg.git /app && \
    pip install --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    cd /app && \
    pip install --no-cache-dir .

WORKDIR /app

COPY --chmod=644 \
    trained_models/resunet_128_128_96_20_09_25/checkpoint_final.pth \
    trained_models/resunet_128_128_96_20_09_25/checkpoint_best.pth \
    trained_models/resunet_128_128_96_20_09_25/dataset_fingerprint.json \
    trained_models/resunet_128_128_96_20_09_25/

ENTRYPOINT [ "/usr/bin/bash", "/app/timelesseg_containerized.sh" ]
