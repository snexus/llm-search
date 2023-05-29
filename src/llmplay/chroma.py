import os
from pathlib import Path
from typing import Callable

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from llama_index import GPTVectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores import ChromaVectorStore
from loguru import logger

from llmplay.nodes import get_nodes_from_documents


class ChromaVS:
    def __init__(
        self,
        persist_folder: str,
        collection_name: str,
        embedding_model_name: str,
        service_context,
        index_id: str = "vector_index",
    ):
        self._embedding_persist_folder = os.path.join(persist_folder, "embeddings")
        self._index_persist_folder = os.path.join(persist_folder, "index")
        self._chroma_client = create_chroma_client(persist_folder=self._embedding_persist_folder)
        self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        self._chroma_collection = self._chroma_client.get_or_create_collection(
            collection_name, embedding_function=self._embedding_function
        )
        self._collection_name = collection_name
        self._index_id = index_id
        self._service_context = service_context

    def get_collection(self, collection_name):
        return self._chroma_client.get_collection(collection_name, embedding_function=self._embedding_function)

    def create_index_from_folder(self, folder_path: str, parser_func: Callable, extension="md"):
        paths = list(Path(folder_path).glob(f"**/*.{extension}*"))

        nodes = get_nodes_from_documents(document_paths=paths, chunk_parser=parser_func)
        storage_context = StorageContext.from_defaults(
            vector_store=ChromaVectorStore(chroma_collection=self._chroma_collection)
        )
        index = GPTVectorStoreIndex(
            nodes=nodes, storage_context=storage_context, service_context=self._service_context
        )
        index.set_index_id(self._index_id)

        logger.info(f"Saving index to {self._index_persist_folder}")
        index.storage_context.persist(self._index_persist_folder)

        logger.info(f"Saving embeddings to {self._embedding_persist_folder}")
        self._chroma_client.persist()
        return index

    def load_index(self):
        sc = StorageContext.from_defaults(
            vector_store=ChromaVectorStore(chroma_collection=self._chroma_collection),
            persist_dir=self._index_persist_folder,
        )
        index = load_index_from_storage(
            storage_context=sc, index_id=self._index_id, service_context=self._service_context
        )
        return index


def create_chroma_client(persist_folder: str):
    return chromadb.Client(
        Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_folder, anonymized_telemetry=False)
    )
