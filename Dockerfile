FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /code

COPY . .

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa -y

RUN apt-get update && \
    apt-get install -y \
    git \
    python3.9 \
    python3.9-dev \
    python3.9-dbg \
    python3.9-distutils \
    python3-pip \
    libglib2.0-0 

RUN ln -s /usr/bin/python3.9 /usr/bin/python
RUN python -m pip install --upgrade -r requirements.txt --no-cache-dir
RUN python -m pip install --upgrade datasets --no-cache-dir
RUN python -m pip install torch \
    torchvision \
    torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu117 \
    --no-cache-dir

ENTRYPOINT [ "./finetune.sh" ]