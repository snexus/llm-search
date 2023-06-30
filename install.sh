#!/bin/bash

python3 -m pip install -U pip

echo "Installing BLAS enabled  llama-cpp-python"
CUDA_PATH="/usr/local/cuda"
CMAKE_ARGS="-DCMAKE_CUDA_COMPILER:PATH=$CUDA_PATH/bin/nvcc -DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python

echo "Installing requirements..."
pip install -e .