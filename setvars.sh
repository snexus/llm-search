#!/bin/bash

echo "Setting arguments for python-llama-cpp to compile with CUBLAS. You can run pip install now."
export CMAKE_ARGS="-DLLAMA_CUBLAS=on"

if [ -n "$1" ]
then
    export CMAKE_ARGS="-DCMAKE_CUDA_COMPILER:PATH=$1/bin/nvcc $CMAKE_ARGS"
fi

export FORCE_CMAKE=1 
python3 -m pip install -U pip