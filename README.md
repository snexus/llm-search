# LLM Playground

## Prerequisites

* Python 3.10, including dev packages (python3-dev on Ubuntu)
* poetry

## Hardware requirements

This example uses the smallest of Dolly 2.0 models, `databricks/dolly-v2-2-8b`, which runs successfully on Nvidia RTX 3060 (12GB).


## Installation

```bash
git clone https://github.com/snexus/llm-playground.git
cd llm-playground

poetry env use 3.10
poetry shell
poetry install
```


## Docker based installation
