
# LLM Search

**WORK IN PROGRESS...**

The goal of this package is to create a convenient experience for LLMs (both OpenAI and locally hosted) to interact with custom documents. 

## Features

* Supported formats
    * `.md` - splits files on a logical level (headings, subheadings, code blocks, etc..). Currently is more advanced than Langchain's built-in parser.
* Generates embeddings from folder of documents and stores in a vector databases:
    * ChromaDB
* Interact with embedding using state-of-the-art LLMs, including local (private):
    * OpenAI (ChatGPT 3.5/4)
    * Databricks Dolly - 3b and 7b variants
    * Mosaic MPT (7b)
    * Falcon (7b)
    * Quantized 4bit GPTQ models (AutoGPTQ)
    * GGML models through LlamaCPP (not for commerical use due to licensing of the base Llama model)
        * WizardLM-1.0 (13B)
        * Nous-Hermes (13B) - https://huggingface.co/TheBloke/Nous-Hermes-13B-GGML
* Other features
    * CLI
    * Ability to load in 8 bit (quantization) to reduce memory footprint on older hardware.
    * Ability to limit context window, to adapt to different requirements of llm models.


## Prerequisites

* Python 3.8+, including dev packages (python3-dev on Ubuntu)


## Virtualenv based installation

```bash
git clone https://github.com/snexus/llm-search.git
cd llm-search

# Create a new environment
python3 -m venv .venv 

# Activate new environment
source .venv/bin/activate

# For CUDA BLAS support for LlamaCpp, together with other depedendencies
./install.sh

# # Or, Install in development mode
# pip install -e .
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

# Install the package in development mode with CUDA BLAS support for LlamaCpp
cd /shared
./install.sh
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

### Example interacting with document database using an OpenAI model

* To interact with OpenAI models, create `.env` in the root directory of the repository, containing OpenAI API key. A template for the `.env` file is provided in `.env_template`

* A code snippet below launches an OpenAI model and limits the context window to 2048 characters. The system will query and provide the most relevant context from the embeddings database, up to a maximum context size.
```bash
python3 cli.py interact llm -f /storage/llm/embeddings -m openai-gpt35 -c /storage/llm/cache  -cs 2048
```

### Example interacting with document database using local (HuggingFace)

* `-q8 1` flag indicates to load the model in 8 bit (quantization). Useful for limited GPU memory.

```bash
python3 cli.py interact llm -f /storage/llm/embeddings -m falcon-7b-instruct -c /storage/llm/cache  -q8 1 -cs 2048
```


## Example interacting with document database using llama-cpp / GGML model

* Currently, GGML models are configured to offload 35 layers to GPU (hard-coded). In the future, there will be an ability to specify it in an external config.

```bash
python3 cli.py interact llm -f /storage/llm/embeddings -m wizardlm-1.0-ggml -c /storage/llm/cache  -cs 2048 --model-path /storage/llm/cache/WizardLM-13B-1.0-GGML/WizardLM-13B-1.0.ggmlv3.q5_K_S.bin
```