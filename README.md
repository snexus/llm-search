[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/snexus/llm-search/blob/main/notebooks/llmsearch_google_colab_demo.ipynb)

# pyLLMSearch - Advanced RAG

[Documentation](https://llm-search.readthedocs.io/en/latest/)

The purpose of this package is to offer a convenient question-answering (RAG) system with a simple YAML-based configuration that enables interaction with multiple collections of local documents. Special attention is given to improvements in various components of the system **in addition to basic LLM-based RAGs** - better document parsing, hybrid search, HyDE enabled search, chat history, deep linking, re-ranking, the ability to customize embeddings, and more. The package is designed to work with custom Large Language Models (LLMs) – whether from OpenAI or installed locally.

## Features

* Supported document formats
    * Build-in parsers:
        * `.md` - Divides files based on logical components such as headings, subheadings, and code blocks. Supports additional features like cleaning image links, adding custom metadata, and more.
        * `.pdf` - MuPDF-based parser.
        * `.docx` - custom parser, supports nested tables.
    * Other common formats are supported by `Unstructured` pre-processor:
        * List of formats see [here](https://unstructured-io.github.io/unstructured/core/partition.html).

* Allows interaction with embedded documents, internally supporting the following models and methods (including locally hosted):
    * OpenAI models (ChatGPT 3.5/4 and Azure OpenAI).
    * HuggingFace models.
    * Llama cpp supported models - for full list see [here](https://github.com/ggerganov/llama.cpp#description).

* Interoperability with LiteLLM + Ollama via OpenAI API, supporting hundreds of different models (see [Model configuration for LiteLLM](sample_templates/llm/litellm.yaml))

* Generates dense embeddings from a folder of documents and stores them in a vector database ([ChromaDB](https://github.com/chroma-core/chroma)).
  * The following embedding models are supported:
    * Hugging Face embeddings.
    * Sentence-transformers-based models, e.g., `multilingual-e5-base`.
    * Instructor-based models, e.g., `instructor-large`.
    * OpenAI embeddings.

* Generates sparse embeddings using SPLADE (https://github.com/naver/splade) to enable hybrid search (sparse + dense).

* An ability to update the embeddings incrementally, without a need to re-index the entire document base.

* Support for table parsing via open-source gmft (https://github.com/conjuncts/gmft) or Azure Document Intelligence.

* Optional support for image parsing using Gemini API.

* Supports the "Retrieve and Re-rank" strategy for semantic search, see [here](https://www.sbert.net/examples/applications/retrieve_rerank/README.html).
    * Besides the originally `ms-marco-MiniLM` cross-encoder, more modern `bge-reranker` is supported.

* Supports HyDE (Hypothetical Document Embeddings) - see [here](https://arxiv.org/pdf/2212.10496.pdf).
    * WARNING: Enabling HyDE (via config OR webapp) can significantly alter the quality of the results. Please make sure to read the paper before enabling.
    * From my own experiments, enabling HyDE significantly boosts quality of the output on a topics where user can't formulate the quesiton using domain specific language of the topic - e.g. when learning new topics.

* Support for multi-querying, inspired by `RAG Fusion` - https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1
    * When multi-querying is turned on (either config or webapp), the original query will be replaced by 3 variants of the same query, allowing to bridge the gap in the terminology and "offer different angles or perspectives" according to the article.

* Supprts optional chat history with question contextualization


* Other features
    * Simple CLI and web interfaces.
    * Deep linking into document sections - jump to an individual PDF page or a header in a markdown file.
    * Ability to save responses to an offline database for future analysis.
    * Experimental API


## Demo

![Demo](media/llmsearch-demo-v2.gif)


## Documentation

[Browse Documentation](https://llm-search.readthedocs.io/en/latest/)
