from pathlib import Path
import shutil
from typing import List

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from loguru import logger

from llmsearch.parsers.nodes import get_documents_from_custom_splitter
from llmsearch.parsers.markdown import markdown_splitter
from llmsearch.parsers.unstructured import UnstructuredSplitter, UnstructuredSplitType
from llmsearch.parsers.pdf import PDFSplitter


class VectorStoreChroma:
    def __init__(self, persist_folder: str):
        self._persist_folder = persist_folder

        # Embeddings model is hard-coded for now
        self._embeddings = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-large"
        )
        # InstructorEmbeddingFunction(model_name="hkunlp/instructor-large")
        # HuggingFaceEmbeddings(model_name=hf_embed_model_name)
        self._splitter_conf = {
            "md": markdown_splitter,
            "pdf": PDFSplitter(chunk_overlap=200).split_document,
            "html": UnstructuredSplitter(
                document_type=UnstructuredSplitType.HTML
            ).split_document,
            "epub": UnstructuredSplitter(
                document_type=UnstructuredSplitType.EPUB
            ).split_document,
        }

    def create_index_from_folder(
        self,
        folder_path: str,
        chunk_size: int,
        extensions: List[str],
        clear_persist_folder=True,
    ):
        p = Path((folder_path))

        all_docs = []
        for extension in extensions:
            logger.info(f"Scanning path for extension: {extension}")
            paths = list(p.glob(f"**/*.{extension}*"))

            splitter = self._splitter_conf[extension]
            docs = get_documents_from_custom_splitter(
                document_paths=paths, splitter_func=splitter, max_size=chunk_size
            )
            logger.info(f"Got {len(docs)} chunks for type: {extension}")
            all_docs.extend(docs)

        if clear_persist_folder:
            pf = Path(self._persist_folder)
            if pf.exists() and pf.is_dir():
                logger.warning(f"Deleting the content of: {pf}")
                shutil.rmtree(pf)

        vectordb = Chroma.from_documents(
            all_docs, self._embeddings, persist_directory=self._persist_folder
        )
        logger.info("Persisting the database..")
        vectordb.persist()

    def load_retriever(self, **kwargs):
        vectordb = Chroma(
            persist_directory=self._persist_folder, embedding_function=self._embeddings
        )
        retriever = vectordb.as_retriever(**kwargs)
        vectordb = None
        return retriever
