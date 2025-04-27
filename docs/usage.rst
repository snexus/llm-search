Create embeddings
-----------------

To create embeddings from documents, follow these steps:

1. Open the command line interface.
2. Run the following command: 


.. code-block:: bash

    llmsearch index create -c /path/to/config.yaml

* The default vector database for dense embeddings is ChromaDB, and default embedding model is `e5-large-v2` (unless specified otherwise using `embedding_model` section such as above), which is known for its high performance. 
* You can find more information about this and other embedding models at `MTEB Leadboard <https://huggingface.co/spaces/mteb/leaderboard>`_.
* In addition to dense embeddings, sparse embedding will be generated in `/path/to/embedding/folder/splade` using SPLADE algorithm. 
* Both dense and sparse embeddings will be used for context search and ranked using an offline re-ranker.


Update embeddings (optional)
----------------------------


When new files are added or existing documents are changed, follow these steps to update the embeddings:

.. code-block:: bash

   llmsearch index update -c /path/to/config.yaml

Executing this command will detect changed or new files (based on MD5 hash) and will incrementally update the changes, without the need to rescan the documents from scratch.

Interact with documents
-----------------------

To interact with the documents using one of the supported LLMs, follow these steps:

1. Open the command line interface.
2. Launch web interface


.. code-block:: bash

    llmsearch interact webapp -c /path/to/config_folder -m /path/to/model_config.yaml


Here `path/to/config/folder` points to a folder of one or more document config files. The tool will scans the configs and allows to switch between them.


API and MCP Server
------------------

To launch FastAPI/MCP server, supply a path semantic search config file in the `FASTAPI_RAG_CONFIG` and path to llm config in `FASTAPI_LLM_CONFIG` environment variable and launch `llmsearchapi` 

.. code-block:: bash

    FASTAPI_RAG_CONFIG="/path/to/config.yaml" FASTAPI_LLM_CONFIG="/path/to/llm.yaml" llmsearchapi

1. The API server will be available at `http://localhost:8000/docs` and can be used to interact with the documents using the LLMs.
2. The MCP server will be available at `http://localhost:8000/mcp` and can be configured via any MCP client, assuming SSE MCP server which should point to the same URL.