import urllib
import uuid
from pathlib import Path
from typing import List

from loguru import logger

from llmsearch.config import Config, Document, DocumentPathSettings
from llmsearch.parsers.doc import docx_splitter
from llmsearch.parsers.markdown import markdown_splitter
from llmsearch.parsers.pdf import PDFSplitter
from llmsearch.parsers.unstructured import (UnstructuredSplitter,
                                            UnstructuredSplitType)


class DocumentSplitter:
    def __init__(self, config: Config) -> None:
        self._splitter_conf = {
            "md": markdown_splitter,
            "docx": docx_splitter,
            "doc": docx_splitter,
            "pdf": PDFSplitter(chunk_overlap=200).split_document,
            "html": UnstructuredSplitter(document_type=UnstructuredSplitType.HTML).split_document,
            "epub": UnstructuredSplitter(document_type=UnstructuredSplitType.EPUB).split_document,
        }
        self.document_path_settings = config.embeddings.document_settings
        self.chunk_sizes = config.embeddings.chunk_sizes

    def split(self) -> List[Document]:
        """Splits documents based on document path settings

        Returns:
            List[Document]: List of documents
        """
        all_docs = []

        for setting in self.document_path_settings:
            passage_prefix = setting.passage_prefix
            docs_path = Path(setting.doc_path)
            exclusion_paths = [str(e) for e in setting.exclude_paths]

            for extension in setting.scan_extensions:
                for chunk_size in self.chunk_sizes: # type: ignore
                    logger.info(f"Scanning path for extension: {extension}")
    
                    # Create a list of document paths to process. Filter out paths in the exclusion list
                    paths = [p for p in list(docs_path.glob(f"**/*.{extension}")) if not self.is_exclusion(p, exclusion_paths)]
    
                    splitter = self._splitter_conf[extension]
    
                    # Get additional parser setting for a given extension, if present
                    additional_parser_settings = setting.additional_parser_settings.get(extension, dict())
    
                    docs = self._get_documents_from_custom_splitter(
                        document_paths=paths, splitter_func=splitter, max_size=chunk_size, passage_prefix=passage_prefix, **additional_parser_settings
                    )
                    logger.info(f"Got {len(docs)} chunks for type: {extension}")
                    all_docs.extend(docs)
            return all_docs

    def is_exclusion(self, path: Path, exclusions: List[str]) -> bool:
        """Checks if path has parent folders in list of exclusions

        Args:
            path (Path): _description_
            exclusions (List[str]): List of exclusion folders

        Returns:
            bool: True if path is in list of exclusions
        """

        exclusion_paths = [Path(p) for p in exclusions]
        for ex_path in exclusion_paths:
            if ex_path in path.parents:
                logger.info(f"Excluding path {path} from documents, as path parent path is excluded.")
                return True
        return False

    def _get_documents_from_custom_splitter(
        self, document_paths: List[Path], splitter_func, max_size, passage_prefix: str, **additional_kwargs
    ) -> List[Document]:
        """Gets list of nodes from a collection of documents

        Examples: https://gpt-index.readthedocs.io/en/stable/guides/primer/usage_pattern.html
        """

        all_docs = []
        if passage_prefix:
            logger.info(f"Will add the following passage prefix: {passage_prefix}")

        for path in document_paths:
            logger.info(f"Processing path using custom splitter: {path}, chunk size: {max_size}")

            # docs_data = splitter_func(text, max_size)
            additional_kwargs.update({"filename": path.name})
            docs_data = splitter_func(path, max_size, **additional_kwargs)
            path = urllib.parse.quote(str(path))
            logger.info(path)
            
            docs = [
                Document(
                    page_content=passage_prefix + d["text"],
                    metadata={**d["metadata"], **{"source": str(path), "chunk_size": max_size, "document_id":str(uuid.uuid1())}},
                )
                for d in docs_data
            ]
            all_docs.extend(docs)

        logger.info(f"Got {len(all_docs)} nodes.")
        return all_docs
