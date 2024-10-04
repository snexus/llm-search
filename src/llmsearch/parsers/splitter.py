import hashlib
import urllib
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from llmsearch.config import Config, Document, PDFTableParser
from llmsearch.parsers.doc import docx_splitter
from llmsearch.parsers.images.generic import get_image_chunks
from llmsearch.parsers.markdown import markdown_splitter
from llmsearch.parsers.pdf import PDFSplitter
from llmsearch.parsers.tables.generic import get_table_chunks
from llmsearch.parsers.unstructured import UnstructuredSplitter

HASH_BLOCKSIZE = 65536


class DocumentSplitter:
    def __init__(self, config: Config) -> None:

        # Custom splitters are configured here
        self._splitter_conf = {
            "md": markdown_splitter,
            "docx": docx_splitter,
            "doc": docx_splitter,
            "pdf": PDFSplitter(chunk_overlap=200).split_document,
        }

        # Fallback splitter unless custom splitter is specified
        self._fallback_splitter = UnstructuredSplitter().split_document

        self.document_path_settings = config.embeddings.document_settings
        self.chunk_sizes = config.embeddings.chunk_sizes
        self.cache_folder = config.cache_folder

    def get_hashes(self) -> pd.DataFrame:
        hash_filename_mappings = []
        logger.info("Scanning hashes of the existing files.")

        for setting in self.document_path_settings:
            docs_path = Path(setting.doc_path)
            exclusion_paths = [str(e) for e in setting.exclude_paths]

            for scan_extension in setting.scan_extensions:
                extension = scan_extension

                # Create a list of document paths to process. Filter out paths in the exclusion list
                paths = [
                    p
                    for p in list(docs_path.glob(f"**/*.{extension}"))
                    if (not self.is_exclusion(p, exclusion_paths)) and (p.is_file())
                ]
                hashes = [
                    {"filename": str(path), "filehash": get_md5_hash(path)}
                    for path in paths
                ]
                hash_filename_mappings.extend(hashes)
        return pd.DataFrame(hash_filename_mappings)

    def split(
        self, restrict_filenames: Optional[List[str]] = None
    ) -> Tuple[List[Document], pd.DataFrame, pd.DataFrame, List[str]]:
        """Splits documents based on document path settings

        Returns:
            List[Document]: List of documents
        """
        all_docs = []

        # Maps between file name and it's hash
        hash_filename_mappings = []

        # Mapping between hash and document ids
        hash_docid_mappings = []

        # Will collect all labels, for persisting later
        all_labels = []

        for setting in self.document_path_settings:
            passage_prefix = setting.passage_prefix
            docs_path = Path(setting.doc_path)
            documents_label = setting.label
            exclusion_paths = [str(e) for e in setting.exclude_paths]


            for scan_extension in setting.scan_extensions:
                extension = scan_extension
                for chunk_size in self.chunk_sizes:  # type: ignore
                    logger.info(f"Scanning path for extension: {extension}")

                    # Create a list of document paths to process. Filter out paths in the exclusion list
                    paths = [
                        p
                        for p in list(docs_path.glob(f"**/*.{extension}"))
                        if not self.is_exclusion(p, exclusion_paths)
                    ]

                    # Used when updating the index, we don't need to parse all files again
                    if restrict_filenames is not None:
                        logger.warning(
                            f"Restrict filenames specificed. Scanning at most {len(restrict_filenames)} files."
                        )
                        paths = [p for p in paths if str(p) in set(restrict_filenames)]

                    # Get splitter unless fallback is specified
                    splitter = self._splitter_conf.get(
                        extension, self._fallback_splitter
                    )

                    # Get additional parser setting for a given extension, if present
                    additional_parser_settings = setting.additional_parser_settings.get(
                        extension, dict()
                    )

                    (
                        docs,
                        hf_mappings,
                        hd_mappings,
                    ) = self._get_documents_from_custom_splitter(
                        document_paths=paths,
                        splitter_func=splitter,
                        max_size=chunk_size,
                        passage_prefix=passage_prefix,
                        label=documents_label,
                        table_splitter=setting.pdf_table_parser,
                        image_parser = setting.pdf_image_parser,
                        **additional_parser_settings,
                    )

                    logger.info(f"Got {len(docs)} chunks for type: {extension}")
                    all_docs.extend(docs)
                    hash_filename_mappings.extend(hf_mappings)
                    hash_docid_mappings.extend(hd_mappings)

                    all_labels+=list(set([d.metadata['label'] for d in docs]))

        all_hash_filename_mappings = pd.DataFrame(hash_filename_mappings)
        all_hash_docid_mappings = pd.concat(hash_docid_mappings, axis=0)

        return all_docs, all_hash_filename_mappings, all_hash_docid_mappings, all_labels

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
                logger.info(
                    f"Excluding path {path} from documents, as path parent path is excluded."
                )
                return True
        return False

    def _get_documents_from_custom_splitter(
        self,
        document_paths: List[Path],
        splitter_func,
        max_size,
        passage_prefix: str,
        label: str,
        table_splitter,
        image_parser,
        **additional_kwargs,
    ) -> Tuple[List[Document], List[dict], List[pd.DataFrame]]:
        """Gets list of nodes from a collection of documents

        Examples: https://gpt-index.readthedocs.io/en/stable/guides/primer/usage_pattern.html
        """

        original_label = label
        all_docs = []

        # Maps between file name and it's hash
        hash_filename_mappings = []

        # Mapping between hash and document ids
        hash_docid_mappings = []

        if passage_prefix:
            logger.info(f"Will add the following passage prefix: {passage_prefix}")

        for path in document_paths:
            logger.info(
                f"Processing path using custom splitter: {path}, chunk size: {max_size}"
            )

            # docs_data = splitter_func(text, max_size)
            filename = str(path)
            additional_kwargs.update({"filename": filename})

            # Placeholders for table and imag chunks, if present
            table_chunks, table_bboxes = list(), dict()
            image_chunks, image_bboxes = list(), dict()

            # If table parsing specific, get back the chunks and add to docs data
            if table_splitter is not None:
                table_chunks, table_bboxes = get_table_chunks(path, max_size, table_splitter, cache_folder = self.cache_folder)

            if image_parser is not None:
                image_chunks, image_bboxes = get_image_chunks(path, max_size, image_parser, cache_folder=self.cache_folder)
            
            # Push table bounding boxes to avoid embedding the same text as in the tables.
            additional_kwargs.update({"table_bboxes": table_bboxes})
            additional_kwargs.update({"image_bboxes": image_bboxes})

            docs_data = splitter_func(path, max_size, **additional_kwargs)
            docs_data += table_chunks
            docs_data += image_chunks


            file_hash = get_md5_hash(path)

            path = urllib.parse.quote(str(path))  # type: ignore
            logger.info(path)

            # If label for a set of documents doesn't exist, set it as document path
            # Assign path to label

            # if not original_label:
            label = str(path)

            docs = [
                Document(
                    page_content=passage_prefix + d["text"],
                    metadata={
                        **d["metadata"],
                        **{
                            "source": str(path),
                            "chunk_size": max_size,
                            "source_chunk_type": d['metadata'].get("source_chunk_type", "text"),
                            "document_id": str(uuid.uuid1()),
                            "label": label,
                        },
                    },
                )
                for d in docs_data
            ]

            for (
                d
            ) in (
                docs
            ):  # 2024/03/14 - unstructured fills page as None for some formats like csv
                if "page" in d.metadata and d.metadata["page"] is None:
                    d.metadata["page"] = -1

            all_docs.extend(docs)

            # Add hash to filename mapping and hash to doc ids mapping
            hash_filename_mappings.append(dict(filename=filename, filehash=file_hash))

            df_hash_docid = (
                pd.DataFrame()
                .assign(docid=[d.metadata["document_id"] for d in docs])
                .assign(filehash=file_hash)
            )

            hash_docid_mappings.append(df_hash_docid)

        logger.info(f"Got {len(all_docs)} nodes.")
        return all_docs, hash_filename_mappings, hash_docid_mappings


def get_md5_hash(file_path: Path) -> str:
    hasher = hashlib.md5()

    with open(file_path, "rb") as file:
        buf = file.read(HASH_BLOCKSIZE)
        while buf:
            hasher.update(buf)
            buf = file.read(HASH_BLOCKSIZE)

    return hasher.hexdigest()



