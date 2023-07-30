from pathlib import Path
from typing import List, Union

import fitz
from langchain.text_splitter import CharacterTextSplitter
from loguru import logger


class PDFSplitter:
    def __init__(self, chunk_overlap: int = 200) -> None:
        """Splits pdf documents

        Args:
            chunk_overlap (int, optional): Overlap in characters. Defaults to 200.
        """
        self.chunk_overlap = chunk_overlap

    def split_document(
        self, document_path: Union[str, Path], max_size: int, **kwargs
    ) -> List[dict]:
        """Splits pdf documents into multiple chunks

        Args:
            document_path (Union[str, Path]): Path to a single PDF document
            max_size (int): Max chunk size in characters.

        Returns:
            List[dict]: List of dictionaries containing parsed documents
        """

        logger.info(f"Partitioning document: {document_path}")

        all_chunks = []
        splitter = CharacterTextSplitter(
            separator="\n",
            keep_separator=True,
            chunk_size=max_size,
            chunk_overlap=self.chunk_overlap,
        )

        doc = fitz.open(document_path)
        current_text = ""
        for page in doc:
            text = page.get_text("block")

            if len(text) > max_size:
                all_chunks.append(
                    {"text": current_text, "metadata": {"page": page.number}}
                )
                chunks = splitter.split_text(text)
                for chunk in chunks:
                    logger.info(
                        f"Flushing chunk. Length: {len(chunk)}, page: {page.number}"
                    )
                    all_chunks.append(
                        {"text": chunk, "metadata": {"page": page.number}}
                    )
                current_text = ""

            elif len(current_text + text) >= max_size:
                if current_text != "":
                    all_chunks.append(
                        {"text": current_text, "metadata": {"page": page.number}}
                    )
                logger.info(
                    f"Flushing chunk. Length: {len(current_text)}, page: {page.number}"
                )
                current_text = text

            # Otherwise, add element's text to current chunk, without re-assigning the page number
            else:
                current_text += text

        # Filter out empty docs
        all_chunks = [
            chunk for chunk in all_chunks if chunk["text"].strip().replace(" ", "")
        ]
        return all_chunks
