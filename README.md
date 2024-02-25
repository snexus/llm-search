[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/snexus/llm-search/blob/main/notebooks/llmsearch_google_colab_demo.ipynb)

# LLM Search

The purpose of this package is to offer a convenient question-answering system with a simple YAML-based configuration that enables interaction with multiple collections of local documents. Special attention is given to improvements in various components of the system **in addition to LLMs** - better document parsing, hybrid search, HyDE enabled search, deep linking, re-ranking, the ability to customize embeddings, and more. The package is designed to work with custom Large Language Models (LLMs) – whether from OpenAI or installed locally.

## Features

* Supported formats
    * Build-in parsers:
        * `.md` - Divides files based on logical components such as headings, subheadings, and code blocks. Supports additional features like cleaning image links, adding custom metadata, and more.
        * `.pdf` - MuPDF-based parser.
        * `.docx` - custom parser, supports nested tables.
    * Other common formats are supported by `Unstructured` pre-processor:
        * List of formats https://unstructured-io.github.io/unstructured/core/partition.html

* Supports multiple collection of documents, and filtering the results by a collection.

* An ability to update the embeddings incrementally, without a need to re-index the entire document base.

* Generates dense embeddings from a folder of documents and stores them in a vector database (ChromaDB).
  * The following embedding models are supported:
    * Huggingface embeddings.
    * Sentence-transformers-based models, e.g., `multilingual-e5-base`.
    * Instructor-based models, e.g., `instructor-large`.

* Generates sparse embeddings using SPLADE (https://github.com/naver/splade) to enable hybrid search (sparse + dense).

* Supports the "Retrieve and Re-rank" strategy for semantic search, see - https://www.sbert.net/examples/applications/retrieve_rerank/README.html.
    * Besides the originally `ms-marco-MiniLM` cross-encoder, more modern `bge-reranker` is supported.

* Supports HyDE (Hypothetical Document Embeddings) - https://arxiv.org/pdf/2212.10496.pdf
    * WARNING: Enabling HyDE (via config OR webapp) can significantly alter the quality of the results. Please make sure to read the paper before enabling.
    * From my own experiments, enabling HyDE significantly boosts quality of the output on a topics where user can't formulate the quesiton using domain specific language of the topic - e.g. when learning new topics.

* Support for multi-querying, inspired by `RAG Fusion` - https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1
    * When multi-querying is turned on (either config or webapp), the original query will be replaced by 3 variants of the same query, allowing to bridge the gap in the terminology and "offer different angles or perspectives" according to the article.

* Allows interaction with embedded documents, internally supporting the following models and methods (including locally hosted):
    * OpenAI models (ChatGPT 3.5/4 and Azure OpenAI).
    * HuggingFace models.
    * Llama cpp supported models - for full list see https://github.com/ggerganov/llama.cpp#description
    * AutoGPTQ models (temporarily disabled due to broken dependencies).

* Interoperability with LiteLLM + Ollama via OpenAI API, supporting hundreds of different models (see [Model configuration for LiteLLM](sample_templates/llm/litellm.yaml))

* Other features
    * Simple CLI and web interfaces.
    * Deep linking into document sections - jump to an individual PDF page or a header in a markdown file.
    * Ability to save responses to an offline database for future analysis.
    * Experimental API


## Demo

![Demo](media/llmsearch-demo-v2.gif)

## Prerequisites

* Tested on Ubuntu 22.04.
* Nvidia GPU is required for embeddings generation and usage of locally hosted models.
* Python 3.10, including dev packages (`python3-dev` on Ubuntu)
* Nvidia CUDA Toolkit (tested with v11.7) - https://developer.nvidia.com/cuda-toolkit
* To interact with OpenAI models, create `.env` in the root directory of the repository, containing OpenAI API key. A template for the `.env` file is provided in `.env_template`
* For parsing `.epub` documents, Pandoc is required - https://pandoc.org/installing.html


## Automatic virtualenv based installation on Linux

```bash
git clone https://github.com/snexus/llm-search.git
cd llm-search

# Create a new environment
python3 -m venv .venv 

# Activate new environment
source .venv/bin/activate

./install_linux.sh
```

## Manual virtualenv based installation

```bash
git clone https://github.com/snexus/llm-search.git
cd llm-search

# Create a new environment
python3 -m venv .venv 

# Activate new environment
source .venv/bin/activate

# Set variables for llama-cpp to compile with CUDA.

# Assuming Nvidia CUDA Toolkit is installed and pointing to `usr/local/cuda` on Ubuntu

source ./setvars.sh 

# Install newest stable torch for CUDA 11.x
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install the package
pip install . # or `pip install -e .` for development
```

# Quickstart

## 1) Create a configuration file for document base


To create a configuration file in YAML format, you can refer to the example template provided in `sample_templates/generic/config_template.yaml`.


The sample configuration file specifies how to load one of the supported locally hosted models, downloaded from Huggingface - 
https://huggingface.co/TheBloke/airoboros-l2-13B-gpt4-1.4.1-GGUF/resolve/main/airoboros-l2-13b-gpt4-1.4.1.Q4_K_M.gguf

As an alternative uncomment the llm section for OpenAI model.

[Sample configuration template](sample_templates/generic/config_template.yaml)


## 2) Create a configuration file for model

To create a configuration file in YAML format, you can refer to the example templates provided in `sample_templates/llm`.

The sample configuration file in [LLamacpp Model Template](sample_templates/llm/llamacpp.yaml)
specifies how to load one of the supported locally hosted models via LLamaCPP, downloaded from Huggingface - 
https://huggingface.co/TheBloke/airoboros-l2-13B-gpt4-1.4.1-GGUF/resolve/main/airoboros-l2-13b-gpt4-1.4.1.Q4_K_M.gguf

As an alternative to other templates provided, for example OpenAI or LiteLLM.



## 3) Create document embeddings

To create embeddings from documents, follow these steps:

1. Open the command line interface.
2. Run the following command: 

```bash
llmsearch index create -c /path/to/config.yaml
```

Based on the example configuration above, executing this command will scan a folder containing markdown and pdf files (`/path/to/docments`) excluding the files in `subfolder1` and `subfolder2` and generate a dense embeddings database in the `/path/to/embedding/folder` directory. Additionally, a local cache folder (`/path/to/cache/folder`) will be utilized to store embedding models, LLM models, and tokenizers.


The default vector database for dense is ChromaDB, and default embedding model is `e5-large-v2` (unless specified otherwise using `embedding_model` section such as above), which is known for its high performance. You can find more information about this and other embedding models at [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard).

In addition to dense embeddings, sparse embedding will be generated in `/path/to/embedding/folder/splade` using SPLADE algorithm. Both dense and sparse embeddings will be used for context search.

## 4) Update document embeddings

When new files are added or existing documents are changed, follow these steps to update the embeddings:

```bash
llmsearch index update -c /path/to/config.yaml
```

Executing this command will detect changed or new files (based on MD5 hash) and will incrementally update only the changes, without the need to rescan the documents from scratch.

## 5) Interact with the documents

To interact with the documents using one of the supported LLMs, follow these steps:

1. Open the command line interface.
2. Run one of the following commands: 

* Web interface:

Scans the configs and allows to switch between them.

```bash
llmsearch interact webapp -c /path/to/config_folder -m sample_templates/llm/llamacpp.yaml
```

* CLI interface:

```bash
llmsearch interact llm -c ./sample_templates/obsidian_conf.yaml -m ./sample_templates/llm/llamacpp.yaml

```

Based on the example configuration provided in the sample configuration file, the following actions will take place:

- The system will load a quantized GGUF model using the LlamaCpp framework. The model file is located at `/storage/llm/cache/airoboros-l2-13b-gpt4-1.4.1.Q4_K_M.gguf`
- Based on the model config, the model will be partially loaded into the GPU (30 layers) and partially into the CPU (remaining layers). The `n_gpu_layers` parameter can be adjusted according to the hardware limitations.
- Additional LlamaCpp specific parameters specified in `model_kwargs` from the `llm->params` section will be passed to the model.
- The system will query the embeddings database using hybrid search algorithm using sparse and dense embeddings. It will provide the most relevant context from different documents, up to a maximum context size of 4096 characters (`max_char_size` in `semantic_search`).
- When displaying paths to relevant documents, the system will replace the part of the path `/home/snexus/projects/knowledge-base` with `obsidian://open?vault=knowledge-base&file=`. This replacement is based on the settings `substring_search` and `substring_replace` in `semantic_search->replace_output_path`. 

## API (experimental)

To launch an api, supply a path config file in the `FASTAPI_LLM_CONFIG` environment variable and launch `llmsearchapi` 

```bash
FASTAPI_LLM_CONFIG="/path/to/config.yaml" llmsearchapi
```
