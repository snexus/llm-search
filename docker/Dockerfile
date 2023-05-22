FROM huggingface/transformers-pytorch-gpu:4.28.1

ENV LANG C.UTF-8
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python3 -m pip --no-cache-dir install" && \
    GIT_CLONE="git clone --depth 10" && \

    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \

    apt-get update && \

# ==================================================================
# tools
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        unzip \
        unrar \
        cmake \
        sudo \
        && \


# ==================================================================
# Libraries
# ------------------------------------------------------------------

    apt-get update && \
    $PIP_INSTALL \
        pandas \
        matplotlib \
        tqdm \
        python-dotenv \
        && \

# ==================================================================
# jupyter
# ------------------------------------------------------------------

    $PIP_INSTALL \
        jupyter \
        && \

# ==================================================================
# jupyterlab
# ------------------------------------------------------------------

    $PIP_INSTALL \
        jupyterlab \
        && \

# ==================================================================
# LLM libraries
# ------------------------------------------------------------------

    $PIP_INSTALL \
        chromadb==0.3.21 \
        llama-index==0.6.9 \
        langchain \
        sentence-transformers==2.2.2 \
        && \
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*
    
EXPOSE 8888 6006