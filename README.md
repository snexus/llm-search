
# LLM Search

**WORK IN PROGRESS...**

The goal of this package is to create a convenient experience for LLMs (both OpenAI and locally hosted) to interact with custom documents. 

## Features

* Supported formats
    * `.md` - splits files on a logical level (headings, subheadings, code blocks, etc..). Currently is more advanced than Langchain's built-in parser.
* Vector databses:
    * ChromaDB
* LLMs:
    * OpenAI (ChatGPT 3.5/4)
    * Databricks Dolly - 3b and 7b variants
    * Mosaic MPT (7b)
    * Falcon (7b)
* Other features
    * CLI
    * Ability to load in 8 bit (quantization) to reduce memory footprint on older hardware.
    * Ability to limit context window, to adapt to different requirements of llm models.


## Prerequisites

* Python 3.10, including dev packages (python3-dev on Ubuntu)
* poetry



## Virtualenv based installation

```bash
git clone https://github.com/snexus/llm-search.git
cd llm-search

# Create a new environment
python3 -m venv .venv 

# Activate new environment
source .venv/bin/activate

# Install in development mode
pip install -e .
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

# Quickstart

## Create embeddings from documents

Scan a folder of markdown files and create an embeddings database.

Assuming documents are stored in `/storage/llm/docs`, the following command will create an embedding database in `/storage/llm/embeddings`. In addition,  `/storage/llm/cache` folder will be used for local cache of embedding models, LLM models and tokenizers.

```bash
cd src/llmsearch
python3 cli.py index create -d /storage/llm/docs -o /storage/llm/embeddings --cache-folder /storage/llm/cache
```

## Interact with the documents using one of the supported LLMs

### Help on available options (including list of inegrated LLMs)

```bash
python3 cli.py index --help
python3 cli.py interact llm --help
```

### Example interacting with document database using OpenAI model

* A code snippet below launches an OpenAI model and limits the context window to 2048 characters. The system will query and provide the most relevant context from the embeddings database, up to a maximum context size.
```bash
python3 cli.py interact llm -f /storage/llm/embeddings -m openai-gpt35 -c /storage/llm/cache  -cs 2048
```

### Example interacting with document database using local (HuggingFace)

* `-q8 1` flag indicates to load the model in 8 bit (quantization). Useful for limited GPU memory.

```bash
python3 cli.py interact llm -f /storage/llm/embeddings -m falcon-7b-instruct -c /storage/llm/cache  -q8 1 -cs 2048
```