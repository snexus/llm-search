# LLM Playground

## Prerequisites

* Python 3.10, including dev packages (python3-dev on Ubuntu)
* poetry

## Hardware requirements

This example uses the smallest of Dolly 2.0 models, `databricks/dolly-v2-2-8b`, which runs successfully on Nvidia RTX 3060 (12GB).


## Installation

```bash
git clone https://github.com/snexus/llm-playground.git
cd llm-playground

# Create a new environment
python3 -m venv .venv 

# Activate new environment
source .venv/bin/activate
# Install the dependencies
pip install -r ./requirements.txt
```


## Docker based installation

### Enable NVIDIA GPU on Docker

* https://linuxhint.com/use-nvidia-gpu-docker-containers-ubuntu-22-04-lts/
* Setup nvidia container toolkit - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit

### Build the container

```bash
docker build -t deepml:latest ./docker
```

### Login interactively (bash) into container

* Run the following script with the argument specifying read-write folder for caching model. This folder will be mounted inside the container under `/storage`

```bash
./run_docker.sh RW_CACHE_FOLDER_NAME
```

# LLM Search

## Create embeddings from documents

Scan a folder of markdown files and create embeddings

```bash
python3 ./cli.py index create -d /storage/llm/docs -o /storage/llm/embeddings --scan-extenstion md
```