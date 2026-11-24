FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

# Python package index for pip. Defaults to PyPI. Pass a mirror URL (without
# credentials) and, if it needs them, a netrc as a BuildKit secret:
#   docker build --build-arg PIP_INDEX_URL=<url> --secret id=cloudsmith_netrc,src=<netrc> .
# The secret is mounted for the install steps only and is never written to a layer.
ARG PIP_INDEX_URL=https://pypi.org/simple

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git curl software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt install -y python3.10 python3.10-distutils
RUN --mount=type=secret,id=cloudsmith_netrc,target=/root/.netrc \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10


RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2 && \
    update-alternatives --auto python3

RUN export CUDA_HOME=/usr/local/cuda/

COPY . /home/llm-finetuning-hub
WORKDIR /home/llm-finetuning-hub
RUN --mount=type=secret,id=cloudsmith_netrc,target=/root/.netrc \
    pip3 install --no-cache-dir -r ./requirements.txt