import urllib
from pathlib import Path
from typing import List

from loguru import logger

from llmsearch.config import Document, DocumentPathSettings
from llmsearch.parsers.markdown import markdown_splitter
from llmsearch.parsers.pdf import PDFSplitter
from llmsearch.parsers.unstructured import UnstructuredSplitter, UnstructuredSplitType


class DocumentSplitter:
    def __init__(self, document_path_settings: List[DocumentPathSettings]) -> None:
        self._splitter_conf = {
            "md": markdown_splitter,
            "pdf": PDFSplitter(chunk_overlap=200).split_document,
            "html": UnstructuredSplitter(document_type=UnstructuredSplitType.HTML).split_document,
            "epub": UnstructuredSplitter(document_type=UnstructuredSplitType.EPUB).split_document,
        }
        self.document_path_settings = document_path_settings

    def split(self) -> List[Document]:
        """Splits documents based on document path settings

        Returns:
            List[Document]: List of documents
        """
        all_docs = []

        for setting in self.document_path_settings:
            docs_path = Path(setting.doc_path)
            exclusion_paths = [str(e) for e in setting.exclude_paths]
            chunk_size = setting.chunk_size

            for extension in setting.scan_extensions:
                logger.info(f"Scanning path for extension: {extension}")

                # Create a list of document paths to process. Filter out paths in the exclusion list
                paths = [p for p in list(docs_path.glob(f"**/*.{extension}")) if not self.is_exclusion(p, exclusion_paths)]

                splitter = self._splitter_conf[extension]

                # Get additional parser setting for a given extension, if present
                additional_parser_settings = setting.additional_parser_settings.get(extension, dict())

                docs = self._get_documents_from_custom_splitter(
                    document_paths=paths, splitter_func=splitter, max_size=chunk_size, **additional_parser_settings
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
        self, document_paths: List[Path], splitter_func, max_size, **additional_kwargs
    ) -> List[Document]:
        """Gets list of nodes from a collection of documents

        Examples: https://gpt-index.readthedocs.io/en/stable/guides/primer/usage_pattern.html
        """

        all_docs = []

        for path in document_paths:
            logger.info(f"Processing path using custom splitter: {path}")

            # docs_data = splitter_func(text, max_size)
            additional_kwargs.update({"filename": path.name})
            docs_data = splitter_func(path, max_size, **additional_kwargs)
            path = urllib.parse.quote(str(path))
            logger.info(path)
            docs = [
                Document(
                    page_content=d["text"],
                    metadata={**d["metadata"], **{"source": str(path)}},
                )
                for d in docs_data
            ]
            all_docs.extend(docs)

        logger.info(f"Got {len(all_docs)} nodes.")
        return all_docs
