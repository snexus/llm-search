
# LLM Search

**WORK IN PROGRESS...**

The goal of this package is to create a convenient experience for LLMs (both OpenAI and locally hosted) to interact with custom documents. 

## Features

* Supported formats
    * `.md` - splits files on a logical level (headings, subheadings, code blocks, etc..). Currently is more advanced than Langchain's built-in parser.
* Generates embeddings from folder of documents and stores in a vector database.
* Interact with embedded documents using state-of-the-art LLMs, supporting the following models and methods (including locally hosted):
    * OpenAI (ChatGPT 3.5/4)
    * HuggingFace models, e.g.
        * Falcon7B, Dolly3B/7B
    * GGML models through LlamaCPP (not for commerical use due to licensing of the base Llama model), e.g.
        * WizardLM-1.0 13B
        * Nous-Hermes 13B
    * AutoGPTQ Models, for example
        * Tulu
* Other features
    * CLI
    * Ability to load in 8 bit (quantization) to reduce memory footprint on older hardware.
    * Ability to limit context window, to adapt to different requirements of llm models.


## Prerequisites

* Nvidia GPU (tested on 10GB VRAM). Less if working with GGML models.
* Linux / WSL
* Python 3.8+, including dev packages (python3-dev on Ubuntu)
* Nvidia CUDA Toolkit - https://developer.nvidia.com/cuda-toolkit
* To interact with OpenAI models, create `.env` in the root directory of the repository, containing OpenAI API key. A template for the `.env` file is provided in `.env_template`


## Virtualenv based installation

```bash
git clone https://github.com/snexus/llm-search.git
cd llm-search

# Create a new environment
python3 -m venv .venv 

# Activate new environment
source .venv/bin/activate

# Install the dependencies
./install.sh

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

## Create a configuration file

Create a configuration .yaml file, check an example template in `sample_templates/config_template.yaml`

Assuming documents are stored in `/storage/docs`, an example confgiration would be the following

```yaml
cache_folder: /storage/llm/cache

embeddings:
  doc_path: /storage/docs
  embeddings_path: /storage/embeddings
  scan_extension: md

semantic_search:
  search_type: mmr
  
  replace_output_path:
    substring_search: storage/docs
    substring_replace: obsidian://open?vault=knowledge-base&file=
  max_char_size: 2048


llm:
  type: llamacpp
  params:
    model_path: /storage/llm/cache/WizardLM-13B-1.0-GGML/WizardLM-13B-1.0.ggmlv3.q5_K_S.bin
    prompt_template: |
          ### Instruction:
          Use the following pieces of context to answer the question at the end. If answer isn't in the context, say that you don't know, don't try to make up an answer.

          ### Context: 
          ---------------
          {context}
          ---------------

          ### Question: {question}
          ### Response:
    model_kwargs:
      n_ctx: 1024
      max_tokens: 512
      temperature: 0.0
      n_gpu_layers: 30
      n_batch: 512
```



## Create embeddings from documents

```bash
cd src/llmsearch
python3 cli.py index create -c config.yaml
```

Scan a folder of markdown files and create an embeddings database.

Based on the configuration above, documents in  `/storage/docs` will be scanned (.md files), and an embedding database will be generated in `/storage/embeddings`. In addition,  `/storage/cache` folder will be used for local cache of embedding models, LLM models and tokenizers.



## Interact with the documents using one of the supported LLMs

```bash
cd src/llmsearch
python3 cli.py interact llm -c config.yaml
```

Based on the example `.yaml` configuration above:

* The system will load a quantized GGML model using LlamaCpp framework. The model is stored in `/storage/llm/cache/WizardLM-13B-1.0-GGML/WizardLM-13B-1.0.ggmlv3.q5_K_S.bin`. 

* The model will be loaded partially into GPU (30 layers, based on `n_gpu_layers` and partially into CPU (rest of the layers). This parameter can be tweaked to suit limitations of a particular hardware.

* Additional LlamaCpp specific parameters specified in `model_kwargs` from `llm->params` will be passed to the model.

* To system will query the embeddings database using Maximal Marginal Relevance algorithm (`mmr` parameter in `semantic_search`) and provide and the most relevant context from different documents, up to a maximum context size of 2048 (`max_char_size` in `semantic_search`)

* When displaying paths to relevant documents, the system will replace a part of the path `storage/llm/docs/` with `obsidian://open?vault=knowledge-base&file=`, based on `substring_search` and `substring_replace` in `semantic_search->replace_output_path` settings.

