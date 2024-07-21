#!/bin/bash

echo "Setting arguments for python-llama-cpp to compile with CUBLAS. You can run pip install now."

# Assuming cuda is pointing to /usr/local/cuda/bin
export PATH=$PATH:/usr/local/cuda/bin  
export CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=$(which nvcc) -DLLAMA_CUBLAS=ON -DLLAMA_CUDA=ON"
export FORCE_CMAKE=1 

python3 -m pip install -U pip