Prerequisites
=============

* Tested on Ubuntu 22.04 and OpenSuse Tumbleweed.
* Nvidia GPU is required for embeddings generation and usage of locally hosted models.
* Python 3.10/3.11, including dev packages (`python3-dev` on Ubuntu)
* Nvidia CUDA Toolkit 
    * Tested with CUDA 11.8 to 12.4 - https://developer.nvidia.com/cuda-toolkit
* To interact with OpenAI models, create `.env` in the root directory of the repository, containing OpenAI API key. A template for the `.env` file is provided in `.env_template`
* For parsing `.epub` documents, Pandoc is required - https://pandoc.org/installing.html



Install Latest Version
======================

.. code-block:: bash
    

    # Create a new environment
    python3 -m venv .venv 

    # Activate new environment
    source .venv/bin/activate

    # Install packages using pip
    pip install pyllmsearch

    # Optional dependencues for Azure parser
    pip install "pyllmsearch[azureparser]"

    # Preferred method (much faster) - install packages using uv
    pip install uv
    uv pip install pyllmsearch




Install from source
===================

.. code-block:: bash

    # Clone the repository

    git clone https://github.com/snexus/llm-search.git
    cd llm-search

    # Optional - Set variables for llama-cpp to compile with CUDA.
    # Assuming Nvidia CUDA Toolkit is installed and pointing to `usr/local/cuda` on Ubuntu

    source ./setvars.sh 

    # Optional - Install newest stable torch for CUDA 11.x
    # pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

    # or for CUDA 12.x version
    # pip3 install torch torchvision

    # Install the package
    pip install . # or `pip install -e .` for development
    
    # For Azure parser, install with optional dependencies
    pip install ."[azureparser]"