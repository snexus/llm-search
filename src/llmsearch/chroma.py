import gc
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import tqdm
from langchain_chroma import Chroma
from loguru import logger

from llmsearch.config import Config
from llmsearch.embeddings import VectorStore, get_embedding_model
from llmsearch.parsers.splitter import Document


class VectorStoreChroma(VectorStore):
    def __init__(self, persist_folder: str, config: Config):
        self._persist_folder = persist_folder
        self._config = config
        self._embeddings = get_embedding_model(config.embeddings.embedding_model)
        self.batch_size = 200  # Limitation of Chromadb (2023/09 v0.4.8) - can add only 41666 documents at once

        self._retriever = None
        self._vectordb = None

    @property
    def retriever(self):
        if self._retriever is None:
            self._retriever = self._load_retriever()
        return self._retriever

    @property
    def vectordb(self):
        if self._vectordb is None:
            self._vectordb = Chroma(
                persist_directory=self._persist_folder,
                embedding_function=self._embeddings,
            )
        return self._vectordb

    def unload(self):
        self._vectordb = None
        self._retriever = None

        gc.collect()

    def create_index_from_documents(
        self,
        all_docs: List[Document],
        clear_persist_folder: bool = True,
    ):
        if clear_persist_folder:
            pf = Path(self._persist_folder)
            if pf.exists() and pf.is_dir():
                logger.warning(f"Deleting the content of: {pf}")
                shutil.rmtree(pf)

        logger.info("Generating and persisting the embeddings..")

        vectordb = None
        for group in tqdm.tqdm(
            chunker(all_docs, size=self.batch_size),
            total=int(len(all_docs) / self.batch_size),
        ):
            ids = [d.metadata["document_id"] for d in group]
            if vectordb is None:
                vectordb = Chroma.from_documents(
                    documents=group,  # type: ignore
                    embedding=self._embeddings,
                    ids=ids,
                    persist_directory=self._persist_folder,  # type: ignore
                )
            else:
                vectordb.add_texts(
                    texts=[doc.page_content for doc in group],
                    embedding=self._embeddings,
                    ids=ids,
                    metadatas=[doc.metadata for doc in group],
                )
        logger.info("Generated embeddings. Persisting...")
        # if vectordb is not None:
            # vectordb.persist()
        vectordb = None

    def _load_retriever(self, **kwargs):
        # vectordb = Chroma(persist_directory=self._persist_folder, embedding_function=self._embeddings)
        return self.vectordb.as_retriever(**kwargs)

    def add_documents(self, docs: List[Document]):
        """Adds new documents to existing vectordb

        Args:
            docs (List[Document]): List of documents
        """

        logger.info(f"Adding embeddings for {len(docs)} documents")
        # vectordb = Chroma(persist_directory=self._persist_folder, embedding_function=self._embeddings)
        for group in tqdm.tqdm(
            chunker(docs, size=self.batch_size), total=int(len(docs) / self.batch_size)
        ):
            ids = [d.metadata["document_id"] for d in group]
            self.vectordb.add_texts(
                texts=[doc.page_content for doc in group],
                embedding=self._embeddings,
                ids=ids,
                metadatas=[doc.metadata for doc in group],
            )
        logger.info("Generated embeddings. Persisting...")
        # self.vectordb.persist()

    def delete_by_id(self, ids: List[str]):
        logger.warning(f"Deleting {len(ids)} chunks.")
        # vectordb = Chroma(persist_directory=self._persist_folder, embedding_function=self._embeddings)
        self.vectordb.delete(ids=ids)
        # self.vectordb.persist()

    def get_documents_by_id(self, document_ids: List[str]) -> List[Document]:
        """Retrieves documents by ids

        Args:
            document_ids (List[str]): list of document ids

        Returns:
            List[Document]: list of documents belonging to document_ids
        """

        results = self.retriever.vectorstore.get(ids=document_ids, include=["metadatas", "documents"])  # type: ignore
        docs = [
            Document(page_content=d, metadata=m)
            for d, m in zip(results["documents"], results["metadatas"])
        ]
        return docs

    def similarity_search_with_relevance_scores(
        self, query: str, k: int, filter: Optional[dict]
    ) -> List[Tuple[Document, float]]:
        # If there are multiple key-value pairs, combine using AND rule - the syntax is chromadb specific
        if isinstance(filter, dict) and len(filter) > 1:
            filter = {"$and": [{key: {"$eq": value}} for key, value in filter.items()]}
            print("Filter = ", filter)

        return self.retriever.vectorstore.similarity_search_with_relevance_scores(
            query, k=self._config.semantic_search.max_k, filter=filter
        )  # type: ignore


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))
