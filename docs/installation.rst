Prerequisites
=============

* Tested on Ubuntu 22.04 and OpenSuse Tumbleweed.
* Nvidia GPU is required for embeddings generation and usage of locally hosted models.
* Python 3.10/3.11, including dev packages (`python3-dev` on Ubuntu)
* Nvidia CUDA Toolkit 
    * Tested with CUDA 11.8 to 12.4 - https://developer.nvidia.com/cuda-toolkit
* To interact with OpenAI models, create `.env` in the root directory of the repository, containing OpenAI API key. A template for the `.env` file is provided in `.env_template`
* For parsing `.epub` documents, Pandoc is required - https://pandoc.org/installing.html
* `uv` - https://github.com/astral-sh/uv#installation



Install Latest Version
======================

.. code-block:: bash
    
    # Create a new environment
    uv venv

    # Activate new environment
    source .venv/bin/activate

    # Optional dependencues for Azure parser
    uv pip install "pyllmsearch[azureparser]"

    # Preferred method (much faster) - install packages using uv
    uv pip install pyllmsearch




Install from source
===================

.. code-block:: bash

    # Clone the repository

    git clone https://github.com/snexus/llm-search.git
    cd llm-search
    # Create a new environment
    uv venv
    # Activate new environment
    source .venv/bin/activate
    # Install packages using uv

    uv sync 

    # Optional - install in the development mode
    uv pip install -e . # or `pip install -e .` for development
    
    # For Azure parser, install with optional dependencies
    uv pip install ."[azureparser]"