import shutil
from pathlib import Path
from typing import List

from langchain.vectorstores import Chroma
from llmsearch.config import EmbeddingModel
from loguru import logger

from llmsearch.parsers.splitter import Document
from llmsearch.embeddings import get_embedding_model

class VectorStoreChroma:
    def __init__(self, persist_folder: str, embeddings_model_config: EmbeddingModel):
        self._persist_folder = persist_folder

        # Embeddings model is hard-coded for now
        # self._embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
        self._embeddings =  get_embedding_model(embeddings_model_config)
        # InstructorEmbeddingFunction(model_name="hkunlp/instructor-large")
        # HuggingFaceEmbeddings(model_name=hf_embed_model_name)

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

        vectordb = Chroma.from_documents(all_docs, self._embeddings, persist_directory=self._persist_folder)
        logger.info("Persisting the database..")
        vectordb.persist()

    def load_retriever(self, **kwargs):
        vectordb = Chroma(persist_directory=self._persist_folder, embedding_function=self._embeddings)
        retriever = vectordb.as_retriever(**kwargs)
        return retriever
