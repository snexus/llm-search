#!/bin/bash

source ./setvars.sh
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install .
