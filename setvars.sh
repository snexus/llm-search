#!/bin/bash

if [ -z "$1" ]
then
    echo "Provide CUDA path as an argument, e.g. /usr/local/cuda"
    exit 1
fi

CUDA_PATH=$1
python3 -m pip install -U pip

echo "Setting arguments for python-llama-cpp to compile with CUBLAS. You can run pip install now."
export CMAKE_ARGS="-DCMAKE_CUDA_COMPILER:PATH=$CUDA_PATH/bin/nvcc -DLLAMA_CUBLAS=on" 
export FORCE_CMAKE=1 

# echo "Installing requirements..."
# pip install -e .