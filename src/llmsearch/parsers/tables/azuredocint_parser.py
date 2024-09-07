from functools import cached_property
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from llmsearch.parsers.tables.generic import (
    GenericParsedTable,
    XMLConverter,
    prepare_and_clean_folder,
)

from llmsearch.parsers.tables.gmft_parser import DocumentHandler, TableDetectorHelper
import pymupdf
from loguru import logger
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import ContentFormat


load_dotenv()

doc_intelligence_endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "")
if not doc_intelligence_endpoint:
    logger.error(
        "Please specify AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=xxx in .env file."
    )
    raise ValueError

doc_intelligence_key = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY", "")
if not doc_intelligence_key:
    logger.error("Please specify AZURE_DOCUMENT_INTELLIGENCE_KEY=xxx in .env file.")
    raise ValueError


class AzureParsedTable(GenericParsedTable):
    def __init__(self, table, default_dpi: int = 72):

        page_number, bbox = self.extract_page_and_bbox(table, dpi=default_dpi)
        logger.info(f"Get bounding box for the table: {bbox}, page: {page_number}")
        super().__init__(page_number, bbox)
        self.table = table

    @cached_property
    def caption(self):
        return self.table.caption

    @property
    def df(self) -> Optional[pd.DataFrame]:
        """Attempts to extract a DataFrame from the cropped table.

        Returns:
            Optional[pd.DataFrame]: The extracted DataFrame or None if extraction fails.

        Raises:
            ExtractionError: If extraction fails, this error will be raised.
        """
        df_temp = pd.DataFrame()

        for cell in self.table["cells"]:
            row_index = cell["rowIndex"]
            col_index = cell["columnIndex"]
            content = cell["content"]
            df_temp.at[row_index, col_index] = content
        return df_temp

    def extract_page_and_bbox(self, table, dpi: int):
        page_number = -1

        top_left_x, top_left_y, bottom_right_x, bottom_right_y = 1e8, 1e8, 0, 0

        for region in table.bounding_regions:
            page_number = region["pageNumber"]
            polygon = region["polygon"]
            tl_x, tl_y, br_x, br_y = polygon[0], polygon[1], polygon[4], polygon[5]

            if tl_x < top_left_x:
                top_left_x = tl_x
            if tl_y < top_left_y:
                top_left_y = tl_y
            if br_x > bottom_right_x:
                bottom_right_x = br_x
            if br_y > bottom_right_y:
                bottom_right_y = br_y

        return page_number, (
            top_left_x * dpi,
            top_left_y * dpi,
            bottom_right_x * dpi,
            bottom_right_y * dpi,
        )

    @property
    def xml(self) -> List[str]:
        """Converts the extracted DataFrame to XML format.

        Returns:
            List[str]: A list of XML strings. Returns an empty list if df extraction failed.
        """
        if self.df is None:
            return []
        return XMLConverter.convert(self.df)


class AzureDocIntelligenceTableParser:
    def __init__(self, fn: Path, temp_folder: Path):
        self.fn = fn

        # Initialie document intelligence client
        self.document_intelligence_client = DocumentIntelligenceClient(
            endpoint=doc_intelligence_endpoint,
            credential=AzureKeyCredential(doc_intelligence_key),
        )
        self.table_pages_extractor = PDFTablePagesExtractor(fn, temp_folder)

    def detect_and_parse_tables(self) -> List[AzureParsedTable]:
        fn, page_mappings = self.table_pages_extractor.extract_table_pages()

        with open(fn, "rb") as f:
            poller = self.document_intelligence_client.begin_analyze_document(
                "prebuilt-layout",
                analyze_request=f,
                content_type="application/octet-stream",
                output_content_format=ContentFormat.MARKDOWN,
            )

        logger.info(f"Calling AzureDocument Intelligence for {fn}")
        result = poller.result()

        out = []
        if result.tables:
            logger.info(f"\tGot {len(result.tables)} table, extracting...")
            out = [AzureParsedTable(table) for table in result.tables]

        return out


class PDFTablePagesExtractor:
    def __init__(self, fn: Path, temp_folder: Path) -> None:
        self.document_handler = DocumentHandler(fn)
        self.table_detector = TableDetectorHelper()
        self.temp_folder = temp_folder
        self.fn = fn

    def _detect_table_pages(self) -> List[int]:
        """Detects pages that contain tables in a pdf file"""

        table_pages = []
        for page in self.document_handler.get_pages():
            tables = self.table_detector.detect_tables(page)
            if tables:
                table_pages.append(page.page_number)
        return table_pages

    def extract_table_pages(self) -> Tuple[Path, Dict[int, int]]:
        # Form an output filename
        prepare_and_clean_folder(self.temp_folder)
        table_pages = self._detect_table_pages()

        page_mappings = {m: p for p, m in zip(table_pages, range(len(table_pages)))}

        if not table_pages:
            logger.info(f"Couldn't find tables in the {self.fn}. Continuing...")

        logger.info(
            f"Found {len(table_pages)} pages with tables in {self.fn}. Extracting..."
        )

        # Form an output filename
        output_fn = self.temp_folder / f"{self.fn.stem}_reduced.pdf"

        with PDFPageExtractor(self.fn) as extractor:
            extractor.extract_save_pages(output_fn, table_pages)

        return output_fn, page_mappings


class PDFPageExtractor:
    def __init__(self, input_filename):
        self.input_filename = input_filename
        self.pdf_document = None

    def __enter__(self):
        self.pdf_document = pymupdf.open(self.input_filename)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pdf_document:
            self.pdf_document.close()

    def extract_save_pages(self, output_filename: Path, pages: List[int]):
        """
        Extracts specified pages from the input PDF and saves them to a new PDF file.

        :param output_filename: The path to the output PDF file.
        :param pages: A list of page numbers to extract (1-indexed).
        :return: True if successful, False otherwise.
        """

        if not self.pdf_document:
            logger.error("PDF document couldn't be open.")
            raise

        try:
            # Create a new PDF document
            new_pdf = pymupdf.open()

            # Adjust page numbers to 0-index and sort them
            pages = sorted(
                page - 1 for page in pages if 1 <= page <= len(self.pdf_document)
            )

            # Extract and add the specified pages
            for page_num in pages:
                new_pdf.insert_pdf(
                    self.pdf_document, from_page=page_num, to_page=page_num
                )

            # Save the new PDF
            new_pdf.save(output_filename)
            new_pdf.close()
            return True

        except Exception as e:
            logger.error(f"An error occurred while creating reduced pdf: {str(e)}")
            raise


if __name__ == "__main__":

    path = Path("/home/snexus/Downloads/Table_Example1-1.pdf")
    parser = AzureDocIntelligenceTableParser(
        fn=path, temp_folder=Path("./azuredoc_temp")
    )
    # ex = PDFTablePagesExtractor(fn = path, temp_folder = Path("./azuredoc_temp"))

    tables = parser.detect_and_parse_tables()
    for table in tables:
        print(table.df)
        print(table.xml)

    # out_fn, page_mappings = ex.extract_table_pages()
    # print(out_fn, page_mappings)
    del parser
