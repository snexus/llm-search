
# LLM Search

**WORK IN PROGRESS...**

The purpose of this package is to create a user-friendly and private question answering system that utilizes a context from local documents. It has the capability to interact with custom Language Models (LLMs), both those hosted by OpenAI or installed locally.

## Features

* Supported formats
    * `.md` - Divides files based on logical components such as headings, subheadings, and code blocks. Currently, this feature is more advanced compared to Langchain's built-in parser.
* Generates embeddings from a folder of documents and stores them in a vector database (ChromaDB).
* Allows interaction with embedded documents using cutting-edge LLMs, supporting the following models and methods (including locally hosted):
    * OpenAI (ChatGPT 3.5/4)
    * HuggingFace models
    * GGML models through LlamaCPP (not for commercial use due to licensing restrictions of the base Llama model), for example:
        * WizardLM-1.0(13B)
        * Nous-Hermes(13B)
    * AutoGPTQ Models
* Other features
    * CLI (Command Line Interface)
    * Supports arbitrary quantization using AutoGPTQ/GGML or 8-bit quantization via bitsandbytes (https://github.com/TimDettmers/bitsandbytes) to reduce memory usage on older hardware. Quantization methods have been tested, and comfortable loading of 13B models on Nvidia RTX 3060 with 10GB VRAM has been achieved.
    * Ability to limit the context window to accommodate different requirements of LLM models.


## Prerequisites

* Nvidia GPU. Tested loading 13B models on RTX3060 with 10GB VRAM, via GGML.
* Linux / WSL.
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


To create a configuration file in YAML format, you can refer to the example template provided in `sample_templates/config_template.yaml`.

For the purpose of this explanation, let's assume that the documents are stored in the `/storage/docs` directory. An example configuration would look like the following:


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



## Creating Document Embeddings

To create embeddings from documents, follow these steps:

1. Open the command line interface.
2. Navigate to the `src/llmsearch` directory.
3. Run the following command: `python3 cli.py index create -c config.yaml`

Executing this command will scan a folder containing markdown files (`/storage/docs`) and generate an embeddings database in the `/storage/embeddings` directory. Additionally, a local cache folder (`/storage/cache`) will be utilized to store embedding models, LLM models, and tokenizers.

The default vector database is ChromaDB, and the embeddings are generated using the `instruct-xlarge` model, which is known for its high performance. You can find more information about this model at [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard).

## Interact with the documents using supported LLMs

To interact with the documents using one of the supported LLMs, follow these steps:

1. Open the command line interface.
2. Navigate to the `src/llmsearch` directory.
3. Run the following command: `python3 cli.py interact llm -c config.yaml`

Based on the example configuration provided in the `.yaml` file, the following actions will take place:

- The system will load a quantized GGML model using the LlamaCpp framework. The model file is located at `/storage/llm/cache/WizardLM-13B-1.0-GGML/WizardLM-13B-1.0.ggmlv3.q5_K_S.bin`.
- The model will be partially loaded into the GPU (30 layers) and partially into the CPU (remaining layers). The `n_gpu_layers` parameter can be adjusted according to the hardware limitations.
- Additional LlamaCpp specific parameters specified in `model_kwargs` from the `llm->params` section will be passed to the model.
- The system will query the embeddings database using the Maximal Marginal Relevance algorithm (`mmr` parameter in `semantic_search`). It will provide the most relevant context from different documents, up to a maximum context size of 2048 characters (`max_char_size` in `semantic_search`).
- When displaying paths to relevant documents, the system will replace the part of the path `/storage/llm/docs/` with `obsidian://open?vault=knowledge-base&file=`. This replacement is based on the settings `substring_search` and `substring_replace` in `semantic_search->replace_output_path`.

