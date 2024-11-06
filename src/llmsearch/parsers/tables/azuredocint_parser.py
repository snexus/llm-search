from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, wait_exponential, stop_after_attempt, before_log, after_log
import os
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pymupdf
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import ContentFormat
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from loguru import logger

from llmsearch.parsers.tables.generic import (
    GenericParsedTable,
    XMLConverter,
    prepare_and_clean_folder,
)
from llmsearch.parsers.tables.gmft_parser import DocumentHandler, TableDetectorHelper

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


def log_attempt_number(retry_state):
    error_message = str(retry_state.outcome.exception())
    logger.error(
        f"API call attempt {retry_state.attempt_number} failed with error: {error_message}. Retrying..."
    )
    # logger.error(f"API call attempt failed. Retrying: {retry_state.attempt_number}...")


class AzureParsedTable(GenericParsedTable):
    def __init__(self, table, page_mapping: dict, default_dpi: int = 72):

        page_number, bbox = self.extract_page_and_bbox(table, dpi=default_dpi)

        page_number = page_mapping[page_number - 1]

        logger.info(f"Get bounding box for the table: {bbox}, page: {page_number}")
        super().__init__(page_number, bbox)
        self.table = table

    @cached_property
    def caption(self) -> str:
        try:
            return self.table.caption["content"]
        except Exception as ex:
            logger.warning("Couldn't extract caption, returning empty string...")
            return ""


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
            content = self.clean_content(cell["content"])
            df_temp.at[row_index, col_index] = content

        # Set first row as an index
        df_temp.columns = df_temp.iloc[0]
        df_temp = df_temp[1:]
        df_temp = df_temp.reset_index(drop=True)

        # Rename duplicate columns ,if present
        df_temp = df_temp.rename(columns=ColumnRenamer(separator="_"))
        return df_temp

    def clean_content(self, content: str) -> str:
        content = content.replace(":unselected:", "").replace(":selected:", "")
        return content

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


class ColumnRenamer:
    def __init__(self, separator=None):
        self.counter = Counter()
        self.sep = separator

    def __call__(self, x):
        index = self.counter[x]  # Counter returns 0 for missing elements
        self.counter[x] = index + 1  # Uses something like `setdefault`
        return f'{x}{self.sep if self.sep and index else ""}{index if index else ""}'


class AzureDocIntelligenceTableParser:
    def __init__(self, fn: Path, cache_folder: Path):
        self.fn = fn

        self.document_intelligence_client = DocumentIntelligenceClient(
            endpoint=doc_intelligence_endpoint,
            credential=AzureKeyCredential(doc_intelligence_key),
        )

        self.table_pages_extractor = PDFTablePagesExtractor(
            fn, cache_folder / "azuredoc_temp"
        )
        self._parsed_tables: Optional[List[AzureParsedTable]] = (
            None  # Cache for parsed tables
        )

    # def detect_and_parse_tables(self) -> List[AzureParsedTable]:
    #     tables = self.table_pages_extractor.extract_table_pages()

    #     all_tables = []

    #     for fn, page_mapping in tables:
    #         with open(fn, "rb") as f:
    #             poller = self.document_intelligence_client.begin_analyze_document(
    #                 "prebuilt-layout",
    #                 analyze_request=f,
    #                 content_type="application/octet-stream",
    #                 output_content_format=ContentFormat.MARKDOWN,
    #             )

    #         logger.info(f"Calling AzureDocument Intelligence for {fn}")
    #         result = poller.result()

    #         out = []
    #         if result.tables:
    #             logger.info(f"\tGot {len(result.tables)} table, extracting...")
    #             out = [AzureParsedTable(table, page_mapping) for table in result.tables]
    #             all_tables += out

    #     return all_tables

    def detect_and_parse_tables(self) -> List[AzureParsedTable]:
        tables = self.table_pages_extractor.extract_table_pages()

        if not tables:
            return []

        with ThreadPoolExecutor(max_workers=min(10, len(tables))) as executor:
            future_to_fn = {
                executor.submit(self._analyze_document, fn, page_mapping): (
                    fn,
                    page_mapping,
                )
                for fn, page_mapping in tables
            }
            all_tables = []
            for future in as_completed(future_to_fn):
                fn, page_mapping = future_to_fn[future]
                try:
                    result = future.result()
                    if result is not None:
                        all_tables += result
                except Exception as e:
                    logger.error(
                        "Exception occurred while analyzing document", exc_info=e
                    )

        return all_tables

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(3),
        after=log_attempt_number,
    )
    def _analyze_document(self, fn: Path, page_mapping):
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
            out = [AzureParsedTable(table, page_mapping) for table in result.tables]

        for table in out:
            if not table.caption:
                logger.info("Couldn't detect caption, trying with custom algorithm...")
                table.caption = self._detect_vertical_caption(table, fn)
        return out

    def _detect_vertical_caption(self, table: AzureParsedTable, fn: Path, n_max: int = 200) -> str:
        
        # Open the document being parsed
        doc = pymupdf.open(fn)
        
        page_number  = table.table.bounding_regions[0]['pageNumber']
        page = doc[page_number-1]
        
        blocks = page.get_text_blocks()

        top_captions, bottom_captions = [], []
        for block in blocks:
            top_cap, bottom_cap = self._detect_caption(table_bbox=table.bbox, block = block)
            top_captions.append(top_cap)
            bottom_captions.append(bottom_cap)


        top_captions = [c for c in top_captions if c.strip()] # clear out empty captions
        bottom_captions = [c for c in bottom_captions if c.strip()]

        all_captions = "\n".join(top_captions)[:n_max] + "\n".join(bottom_captions)[:n_max]
        caption =  all_captions

        logger.debug(f"Custom algo caption: {caption}")
        return caption


    def _detect_caption(self, table_bbox: Tuple[float, float, float, float], block: Tuple[float, float, float, float, str], max_abs_dist: float = 3.5) -> Tuple[str,str]:
        x1, y1, x2, y2 = block[:4]
        text = block[4]

        # Block in PyMupdf can consist of multiple lines of text
        n_lines = text.count('\n') + 1


        normalized_dist = 1000
        top_caption, bottom_caption = "", ""

        # Take care of captions above the table
        if y2 < table_bbox[1]: # block in question is above the table
            # Normalized distance = how many word "lines" this current sentence is from the table
            normalized_dist =  (y2-table_bbox[1])/((y2-y1) / n_lines)
            if abs(normalized_dist) < max_abs_dist:
                top_caption = block[4]
        
        # Take care of captions below the table
        elif y1 > table_bbox[3]: # block in question is below the table
            normalized_dist =  (y1-table_bbox[3])/((y2-y1)/n_lines)
            if abs(normalized_dist) < max_abs_dist:
                bottom_caption = block[4]
        return top_caption, bottom_caption


    @property
    def parsed_tables(self) -> List[AzureParsedTable]:
        """Lazy-loads the parsed tables when requested.

        Returns:
            List[AzureParsedTable]: A list of parsed tables from the document.
        """
        if self._parsed_tables is None:
            self._parsed_tables = (
                self.detect_and_parse_tables()
            )  # Detect and parse tables if not done already
        return self._parsed_tables


class PDFTablePagesExtractor:
    def __init__(self, fn: Path, temp_folder: Path, max_pages: int = 2) -> None:
        self.document_handler = DocumentHandler(fn)
        self.table_detector = TableDetectorHelper()
        self.temp_folder = temp_folder
        self.fn = fn

        # Azure free tier limits maximum number of pages to two
        self.max_pages = max_pages

    def _detect_table_pages(self) -> List[int]:
        """Detects pages that contain tables in a pdf file"""

        table_pages = []
        for page in self.document_handler.get_pages():
            tables = self.table_detector.detect_tables(page)
            if tables:
                table_pages.append(page.page_number)
        return table_pages

    def extract_table_pages(self) -> List[Tuple[Path, Dict[int, int]]]:
        # Form an output filename
        prepare_and_clean_folder(self.temp_folder)
        table_pages = self._detect_table_pages()

        if not table_pages:
            logger.info(f"Couldn't find tables in the {self.fn}. Continuing...")

        logger.info(
            f"Found {len(table_pages)} pages with tables in {self.fn}. Extracting..."
        )
        outputs = []
        for batch_n, table_page_batch in enumerate(
            iterate_in_batches(table_pages, batch_size=self.max_pages)
        ):

            # Form an output filename
            page_mappings = {
                m: p for p, m in zip(table_page_batch, range(len(table_page_batch)))
            }
            output_fn = self.temp_folder / f"{self.fn.stem}_reduced_{batch_n}.pdf"

            with PDFPageExtractor(self.fn) as extractor:
                extractor.extract_save_pages(output_fn, table_page_batch)

            outputs.append((output_fn, page_mappings))
        return outputs


def iterate_in_batches(lst: List, batch_size: int):
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


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
            # pages = sorted(
            #     page - 1 for page in pages if 1 <= page <= len(self.pdf_document)
            # )

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

    path = Path("/home/snexus/Downloads/Table_Example2.pdf")
    parser = AzureDocIntelligenceTableParser(fn=path, cache_folder=Path("."))
    # ex = PDFTablePagesExtractor(fn = path, temp_folder = Path("./azuredoc_temp"))

    tables = parser.parsed_tables
    for table in tables:
        print(table.df)
        print(table.xml)
        print(table.bbox)
        print(table.page_num)
        print(table.caption)

    # out_fn, page_mappings = ex.extract_table_pages()
    # print(out_fn, page_mappings)
    del parser
