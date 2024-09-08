from pathlib import Path
from typing import Dict, List, Tuple, Union

# import fitz
import pymupdf
from langchain_text_splitters import CharacterTextSplitter
from loguru import logger

from llmsearch.parsers.tables.generic import do_boxes_intersect


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

        # Get table bounding boxes if present
        table_bboxes = kwargs.get("table_bboxes", None)

        # Get image bounding boxes if present
        image_bboxes = kwargs.get("image_bboxes", None)

        logger.info(f"Got table bboxes {table_bboxes}")
        logger.info(f"Got image bboxes {image_bboxes}")

        all_chunks = []
        splitter = CharacterTextSplitter(
            separator="\n",
            keep_separator=True,
            chunk_size=max_size,
            chunk_overlap=self.chunk_overlap,
        )

        doc = pymupdf.open(document_path)
        current_text = ""
        for page in doc:
            # text = page.get_text("block")
            blocks = page.get_text_blocks()

            filter_bboxes = table_bboxes.get(page.number, list()) + image_bboxes.get(page.number, list())
            if filter_bboxes:
                text = filter_blocks(blocks, filter_bboxes=filter_bboxes, page_num=page.number)
            else:
                text = "".join([b[4] for b in blocks])

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




def filter_blocks(blocks: List[Tuple[float, float, float, float, str]], 
                  filter_bboxes:  List[Tuple[float, float, float, float]], 
                  page_num: int) -> str:
    """
    Filter text blocks based on their spatial relation to table bounding boxes.

    This function checks each text block against the bounding boxes of tables
    on a specified page. If a block intersects with a table bounding box,
    that block will be excluded from the returned concatenated string.

    Parameters:
    - blocks (List[Tuple[float, float, float, float, str]]): A list of blocks, 
      where each block is represented as a tuple containing the coordinates 
      (x_min, y_min, x_max, y_max) and the corresponding text (str).
    - table_bboxes (Dict[int, Tuple[float, float, float, float]]): A dictionary 
      where the key is the page number (int), and the value is a tuple 
      representing the coordinates of the bounding boxes of the tables (x_min, 
      y_min, x_max, y_max).
    - page_num (int): The page number for which to filter the blocks.

    Returns:
    - str: A concatenated string of text from blocks that do not intersect
      with any table bounding box on the specified page. If there are no 
      tables, all block texts will be concatenated and returned.

    Example:
    >>> blocks = [(100, 100, 200, 200, "Block 1"),
                  (150, 150, 250, 250, "Block 2")]
    >>> filter_bboxes = [(120, 120, 180, 180)]
    >>> print(filter_blocks(blocks, filter_bboxes, 1))
    "Block 1" (only Block 1 is not intersecting with the table)
    """
    
    # Extract tables belonging to a specific page
    logger.info(f"Page: {page_num}")
    page_table_bboxes = filter_bboxes
    logger.info(f"Got page table bboxes: {page_table_bboxes}")

    if not page_table_bboxes:
        return "".join([b[4] for b in blocks])  # Concat all the text and return if there are no tables
    

    s = ""
    for block in blocks:
        block_bbox = (block[0], block[1], block[2], block[3])
        
        # Flag to determine if we should skip the current block
        skip_block = False
        
        for filter_bbox in page_table_bboxes:
            if do_boxes_intersect(filter_bbox, block_bbox):
                # We found an intersection, set the flag and break the inner loop
                skip_block = True
                # print(f"SKipping block: {block}")
                break  # Exit the inner loop

        if skip_block:
            continue  # Skip the current block and continue with the next block

        # Add block text or other processing if no intersection was found
        s += block[4]  # Assuming block[4] contains the text
    

    return s  # Return the concatenated string