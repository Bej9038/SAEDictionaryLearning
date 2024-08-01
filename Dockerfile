FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

RUN pip install tqdm einops audiogen-agc julius

RUN mkdir -p /workspace
WORKDIR /workspace

COPY main.py /workspace/main.py
COPY utils.py /workspace/utils.py

#COPY D:/Dictionary Learning Dataset /workspace/dataset/
