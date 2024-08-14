from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from loguru import logger

from abc import ABC, abstractmethod

from llmsearch.config import PDFTableParser


class GenericParsedTable(ABC):
    def __init__(self, page_number: int, bbox: Tuple[float, float, float, float]):
        self.page_num = page_number  # Common field
        self.bbox = bbox

    @property
    @abstractmethod
    def df(self) -> pd.DataFrame:
        """Returns Pandas DF corresponding to a table"""
        pass

    @property
    @abstractmethod
    def caption(self) -> str:
        """Returns caption of the table"""
        pass

    @property
    @abstractmethod
    def xml(self) -> List[str]:
        """Returns xml representation of the table"""
        pass


def pandas_df_to_xml(df: pd.DataFrame) -> List[str]:
    """Converts Pandas df to a simplified xml representation digestible by LLMs

    Args:
        df (pd.DataFrame): Pandas df

    Returns:
        str: List of xml row strings representing the dataframe
    """

    def func(row):
        xml = ["<row>"]
        for field in row.index:
            xml.append('  <col name="{0}">{1}</col>'.format(field, row[field]))
        xml.append("</row>")
        return "\n".join(xml)

    items = df.apply(func, axis=1).tolist()
    return items
    # return "\n".join(items)


def pdf_table_splitter(
    parsed_table: GenericParsedTable,
    max_size: int,
    include_caption: bool = True,
    max_caption_size_ratio: int = 4,
):

    xml_elements = parsed_table.xml
    caption = parsed_table.caption
    metadata = {"page": parsed_table.page_num, "source_chunk_type": "table"}

    all_chunks = []

    # If caption is too long, trim it down, so there is some space for actual data
    if len(caption) > max_size / max_caption_size_ratio:
        logger.warning(
            "Caption is too large compared to max char size, trimming down..."
        )
        caption = caption[: int(max_size / max_caption_size_ratio)]

    header = "```xml table:\n"
    if include_caption and caption:
        header = f"Table below contains information about: {caption}\n" + header

    footer = f"```"

    current_text = header
    for el in xml_elements:

        # If new element is too big, trim it (shouldn't happen)
        if len(el) > max_size:
            logger.warning(
                "xml element is larger than allowed max char size. Flushing.."
            )
            # el = el[:max_size-len(header)-3]
            all_chunks.append(
                {"text": current_text + el + footer, "metadata": metadata}
            )
            current_text = header

        # if current text is already large and doesn't fit the new element, flush it
        elif len(current_text + el) >= max_size:
            all_chunks.append({"text": current_text + footer, "metadata": metadata})
            current_text = header + el + "\n"
        else:
            current_text += el + "\n"

    # Flush the last chunk
    all_chunks.append({"text": current_text + footer, "metadata": metadata})
    return all_chunks

def boxes_intersect(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> bool:
    """
    Check if two bounding boxes intersect.

    Parameters:
    box1: Tuple (x1_min, y1_min, x1_max, y1_max)
    box2: Tuple (x2_min, y2_min, x2_max, y2_max)

    Returns:
    True if the boxes intersect, False otherwise.
    """

    # Unpack the box coordinates
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Check for non-intersection
    if x1_max < x2_min or x2_max < x1_min:
        return False
    if y1_max < y2_min or y2_max < y1_min:
        return False

    # If none of the non-intersection conditions are met, they must intersect
    return True

def get_table_chunks(
    path: Path, max_size: int, table_parser: PDFTableParser, format_extensions = ("pdf",)
) -> Tuple[List[dict], Dict[int, List[Tuple[float]]]]:
    """Parses tables from the document using specified table_splitter

    Args:
        path (Path): document path
        max_size (int): Maximum chunk size to split by
        table_splitter (PDFTableParser): name of the table splitter
    """

    table_chunks = []
    extension = str(path).strip("/")[-3:]
    if extension not in  format_extensions:
        logger.info(f"Format {extension} doesn't support table parsing..Skipping..")
        return list(), dict()

    if table_parser is PDFTableParser.GMFT:
        from llmsearch.parsers.tables.gmft_parser import GMFTParser
        parser = GMFTParser(fn=path)
        splitter = pdf_table_splitter
    else:
        raise TypeError(f"Unknown table parser: {table_parser}")

    logger.info("Parsing tables..")

    parsed_tables = parser.parsed_tables

    logger.info(f"Parsed {len(parsed_tables)} tables. Chunking...")
    for parsed_table in parsed_tables:
        table_chunks += splitter(parsed_table, max_size=max_size)

    # Extract tables bounding boxes and store in a convenient data structure.
    table_bboxes = defaultdict(list)
    for table in parsed_tables:
        table_bboxes[table.page_num].append(table.bbox)

    return table_chunks, table_bboxes