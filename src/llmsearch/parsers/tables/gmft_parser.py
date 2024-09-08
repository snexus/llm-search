from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd
from gmft import AutoTableFormatter, CroppedTable, TableDetector
from gmft.pdf_bindings import PyPDFium2Document
from loguru import logger

from llmsearch.parsers.tables.generic import (GenericParsedTable, XMLConverter,
                                              pandas_df_to_xml)


class ExtractionError(Exception):
    """Custom exception for extraction failures."""

    pass


@dataclass
class PageTables:
    """Holds cropped tables extracted from a specific page of a document."""

    page_num: int
    cropped_tables: List[CroppedTable]

    @property
    def n_tables(self) -> int:
        """Returns the number of cropped tables extracted from the page."""
        return len(self.cropped_tables)


class TableFormatterSingleton:
    """Singleton class for managing a single instance of AutoTableFormatter."""

    _instance: Optional["TableFormatterSingleton"] = None
    formatter = None

    def __new__(cls, *args, **kwargs):
        """Creates a new instance if one does not already exist."""
        if cls._instance is None:
            logger.info("Initializing AutoTableFormatter...")
            cls._instance = super().__new__(cls)
            cls._instance.formatter = AutoTableFormatter()
        return cls._instance


class GMFTParsedTable(GenericParsedTable):
    """Represents a parsed table with its metadata and data extraction logic."""

    def __init__(
        self, table: CroppedTable, page_num: int, formatter: AutoTableFormatter
    ) -> None:
        """Initializes the parsed table with a cropped table, page number, and formatter.

        Args:
            table (CroppedTable): The cropped table to parse.
            page_num (int): The page number where the table is found.
            formatter (AutoTableFormatter): The formatter to be used for extraction.
        """
        super().__init__(page_number=page_num, bbox=table.bbox)
        self._table = table  # Store the cropped table
        self.failed = False  # Track extraction failures
        self.formatter = formatter  # Formatter for extracting data

    @cached_property
    def _captions(self) -> List[str]:
        """Caches and returns a list of non-empty captions from the table."""
        return [c for c in self._table.captions() if c.strip()]

    @cached_property
    def caption(self) -> str:
        """Returns a unique string of all captions, combined into one."""
        return "\n".join(set(self._captions))

    @property
    def df(self) -> Optional[pd.DataFrame]:
        """Attempts to extract a DataFrame from the cropped table.

        Returns:
            Optional[pd.DataFrame]: The extracted DataFrame or None if extraction fails.

        Raises:
            ExtractionError: If extraction fails, this error will be raised.
        """
        ft = self.formatter.extract(
            self._table
        )  # Use the formatter to extract the table
        try:
            return ft.df()  # Return the DataFrame
        except ValueError as ex:
            logger.error(f"Couldn't extract df on page {self.page_num}: {str(ex)}")
            self.failed = True
            return None
            # raise ExtractionError(f"Extraction failed on page {self.page_num}")

    @property
    def xml(self) -> List[str]:
        """Converts the extracted DataFrame to XML format.

        Returns:
            List[str]: A list of XML strings. Returns an empty list if df extraction failed.
        """
        if self.df is None:
            return []
        return XMLConverter.convert(self.df)

class DocumentHandler:
    """Handles loading a PDF document and providing access to its pages."""

    def __init__(self, path: Path):
        """Initializes the DocumentHandler with a path to a PDF.

        Args:
            path (Path): The file path to the PDF document.
        """
        self.doc = PyPDFium2Document(path)  # Load the document using PyPDFium2

    def get_pages(self) -> Any:
        """Returns an iterable of pages from the loaded document."""
        return self.doc


class TableDetectorHelper:
    """Facilitates detection of tables within document pages."""

    def __init__(self):
        """Initializes the TableDetector to find tables."""
        self.detector = TableDetector()

    def detect_tables(self, page: Any) -> List[CroppedTable]:
        """Detects and returns cropped tables from a given page.

        Args:
            page (Any): The page from which to detect tables.

        Returns:
            List[CroppedTable]: A list of detected cropped tables.
        """
        return self.detector.extract(page)


class TableParser:
    """Parses cropped tables into GMFTParsedTable objects."""

    def __init__(self, formatter: AutoTableFormatter):
        """Initializes the TableParser with a formatter.

        Args:
            formatter (AutoTableFormatter): Formatter used for parsing tables.
        """
        self.formatter = formatter

    def parse(self, cropped_table: CroppedTable, page_num: int) -> GMFTParsedTable:
        """Parses a cropped table into a GMFTParsedTable instance.

        Args:
            cropped_table (CroppedTable): The cropped table to parse.
            page_num (int): The page number where the table is found.

        Returns:
            GMFTParsedTable: An instance of GMFTParsedTable containing the parsed data.
        """
        return GMFTParsedTable(cropped_table, page_num, self.formatter)


class GMFTParser:
    """Main class for handling the parsing of tables from a PDF document."""

    def __init__(self, fn: Path, **kwargs) -> None:
        """Initializes the parser with a PDF file path and prepares components.

        Args:
            fn (Path): The file path to the PDF document.
        """
        self.fn = fn
        self.document_handler = DocumentHandler(fn)  # Load the document
        self.formatter = TableFormatterSingleton().formatter  # Get the formatter
        self.table_detector = TableDetectorHelper()  # Initialize table detector
        self.table_parser = TableParser(self.formatter)  # Initialize table parser
        self._parsed_tables: Optional[List[GMFTParsedTable]] = (
            None  # Cache for parsed tables
        )

    def detect_and_parse_tables(self) -> List[GMFTParsedTable]:
        """Detects and parses tables from the PDF document.

        Returns:
            List[GMFTParsedTable]: A list of parsed tables.
        """
        logger.info("Detecting and parsing tables...")
        detected_tables = []

        # Iterate through the pages in the document
        for page in self.document_handler.get_pages():
            cropped_tables = self.table_detector.detect_tables(
                page
            )  # Detect tables on the page
            # Parse each cropped table found on the page
            for cropped_table in cropped_tables:
                parsed_table = self.table_parser.parse(cropped_table, page.page_number)
                detected_tables.append(parsed_table)  # Store the parsed table

        return detected_tables

    @property
    def parsed_tables(self) -> List[GMFTParsedTable]:
        """Lazy-loads the parsed tables when requested.

        Returns:
            List[GMFTParsedTable]: A list of parsed tables from the document.
        """
        if self._parsed_tables is None:
            self._parsed_tables = (
                self.detect_and_parse_tables()
            )  # Detect and parse tables if not done already
        return self._parsed_tables


if __name__ == "__main__":
    # fn = Path("/home/snexus/Downloads/ws90.pdf")
    # fn = Path("/home/snexus/Downloads/SSRN-id2741701.pdf")
    # fn = Path("/home/snexus/Downloads/ws90.pdf")
    # fn = Path("/home/snexus/projects/azure-doc-int/notebooks/data/Table_Example1-1.pdf")

    fn = Path("/home/snexus/Downloads/Table_Example1-1.pdf")
    parser = GMFTParser(fn=fn)
    for p in parser.parsed_tables:
        print("-------------")
        print(p.page_num)
        print(p.caption)
        print(p.bbox)
        # print("\n".join(p.xml))

    # chunks = pdf_table_splitter(parsed_table=parser.parsed_tables[7], max_size = 1024)
    # for chunk in chunks:
    # print("\n=========== CHUNK START =============\n")
    # print(chunk['text'])
    # # print(chunks)
    del parser
