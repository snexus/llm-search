import importlib
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from loguru import logger

from llmsearch.config import PDFTableParser

# Define a mapping of PDFImageParser to corresponding analyzer classes and config
PARSER_MAPPING: Dict[PDFTableParser, Any] = {
    PDFTableParser.GMFT: {
        "import_path": "llmsearch.parsers.tables.gmft_parser",  # Import path for lazy loading
        "class_name": "GMFTParser",
        "params": {},
    },
    
    PDFTableParser.AZUREDOC: {
        "import_path": "llmsearch.parsers.tables.azuredocint_parser",  # Import path for lazy loading
        "class_name": "AzureDocIntelligenceTableParser",
        "params": {},
    },
    # Add more analyzers here as needed
    # PDFImageParser.ANOTHER_TYPE: {'import_path': 'another.module.path', 'class_name': 'AnotherAnalyzer', 'params': {'param1': value1, 'param2': value2}},
}


def create_table_parser(table_parser: PDFTableParser, filename: Path, cache_folder: Path):
    parser_info = PARSER_MAPPING.get(table_parser)

    if parser_info is None:
        raise ValueError(f"Unsupported table parser type: {table_parser}")

    # Lazy load the module
    module = importlib.import_module(parser_info["import_path"])
    parser_class = getattr(module, parser_info["class_name"])
    additional_parser_params = parser_info["params"]

    return parser_class(fn = filename, cache_folder = cache_folder, **additional_parser_params)


class GenericParsedTable(ABC):
    def __init__(self, page_number: int, bbox: Tuple[float, float, float, float]):
        self.page_num = page_number  # Common field
        self.bbox = bbox

    @property
    @abstractmethod
    def df(self) -> pd.DataFrame:
        """Returns a Pandas DataFrame corresponding to a table."""
        pass

    @property
    @abstractmethod
    def caption(self) -> str:
        """Returns the caption of the table."""
        pass

    @property
    @abstractmethod
    def xml(self) -> List[str]:
        """Returns xml representaiton of the table"""
        pass
    

class XMLConverter:
    """Converts Pandas DataFrames to XML format."""
    
    @staticmethod
    def convert(df: pd.DataFrame) -> List[str]:
        """Converts a DataFrame to a list of XML strings.

        Args:
            df (pd.DataFrame): The DataFrame to convert.

        Returns:
            List[str]: A list of XML strings representing the DataFrame.
        """
        return pandas_df_to_xml(df)

def pandas_df_to_xml(df: pd.DataFrame) -> List[str]:
    """Converts a Pandas DataFrame to a simplified XML representation.

    Args:
        df (pd.DataFrame): The DataFrame to convert.

    Returns:
        List[str]: List of XML row strings representing the DataFrame.
    """

    def func(row):
        xml = ["<row>"]
        for field in row.index:
            xml.append(f'  <col name="{field}">{row[field]}</col>')
        xml.append("</row>")
        return "\n".join(xml)

    items = df.apply(func, axis=1).tolist()
    return items


def pdf_table_splitter(
    parsed_table: GenericParsedTable,
    max_size: int,
    include_caption: bool = True,
    max_caption_size_ratio: int = 4,
) -> List[Dict[str, Any]]:
    """Splits a parsed table into manageable chunks.

    Args:
        parsed_table (GenericParsedTable): The parsed table instance.
        max_size (int): Maximum size for each chunk.
        include_caption (bool): Whether to include the table caption.
        max_caption_size_ratio (int): Ratio to determine allowable caption size.

    Returns:
        List[Dict[str, Any]]: List of text chunks with metadata.
    """

    xml_elements = parsed_table.xml
    caption = parsed_table.caption
    metadata = {"page": parsed_table.page_num, "source_chunk_type": "table"}
    all_chunks = []

    # Trim caption if it's too long
    if len(caption) > max_size / max_caption_size_ratio:
        logger.warning(
            "Caption is too large compared to max char size, trimming down..."
        )
        caption = caption[: int(max_size / max_caption_size_ratio)]

    header = "```xml table:\n"
    if include_caption and caption:
        header = f"Table below contains information about: {caption}\n" + header

    footer = "```"
    current_text = header
    for el in xml_elements:
        if len(el) > max_size:
            logger.warning(
                "XML element is larger than allowed max char size. Flushing.."
            )
            all_chunks.append({"text": current_text + footer, "metadata": metadata})
            all_chunks.append({"text": header + el + footer, "metadata": metadata})
            current_text = header
        elif len(current_text + el) >= max_size:
            if current_text != header:
                all_chunks.append({"text": current_text + footer, "metadata": metadata})
            current_text = header + el + "\n"
        else:
            current_text += el + "\n"

    # Flush the last chunk
    all_chunks.append({"text": current_text + footer, "metadata": metadata})
    return all_chunks


def do_boxes_intersect(
    box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]
) -> bool:
    """Check if two bounding boxes intersect.

    Args:
        box1 (Tuple[float, float, float, float]): First bounding box.
        box2 (Tuple[float, float, float, float]): Second bounding box.

    Returns:
        bool: True if the boxes intersect, False otherwise.
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    return not (
        x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min
    )


def get_table_chunks(
    path: Path,
    max_size: int,
    table_parser: PDFTableParser,
    cache_folder, 
    format_extensions: Tuple[str, ...] = (".pdf",),
) -> Tuple[List[Dict[str, Any]], Dict[int, List[Tuple[float, float, float, float]]]]:
    """Parses tables from a document and splits them into chunks.

    Args:
        path (Path): Document path.
        max_size (int): Maximum chunk size to split by.
        table_parser (PDFTableParser): Table parser to use.
        format_extensions (Tuple[str, ...]): Supported file formats for parsing.

    Returns:
        Tuple[List[Dict[str, Any]], Dict[int, List[Tuple[float, float, float, float]]]]:
            A tuple with the list of table chunks and a dictionary of bounding boxes.
    """
    table_chunks = []
    extension = path.suffix.lower()
    if extension not in format_extensions:
        logger.info(f"Format {extension} doesn't support table parsing..Skipping..")
        return [], {}

    parser = create_table_parser(table_parser, filename=path, cache_folder = cache_folder)

    logger.info("Parsing tables..")
    parsed_tables = parser.parsed_tables

    logger.info(f"Parsed {len(parsed_tables)} tables. Chunking...")
    for parsed_table in parsed_tables:
        table_chunks.extend(pdf_table_splitter(parsed_table, max_size=max_size))

    # Extract bounding boxes
    table_bboxes = defaultdict(list)
    for table in parsed_tables:
        table_bboxes[table.page_num].append(table.bbox)

    return table_chunks, table_bboxes

def prepare_and_clean_folder(temp_folder: Path):
    if not temp_folder.exists():
        temp_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created folder: {temp_folder}")
    else:
        for file in temp_folder.iterdir():
            if file.is_file():
                file.unlink()
                logger.debug(f"Deleted file: {file}")