from functools import cached_property
import pandas as pd
from typing import Any, List, Optional, Tuple
from gmft.pdf_bindings import PyPDFium2Document
from gmft import (
    CroppedTable,
    TableDetector,
    AutoFormatConfig,
    AutoTableFormatter,
)
from pathlib import Path
from loguru import logger
from dataclasses import dataclass

from llmsearch.parsers.tables.generic import pandas_df_to_xml, GenericParsedTable, pdf_table_splitter


# logger.info("Creating AutoTableFormatter")
# formatter = AutoTableFormatter() # Create singleton

class TableFormatterSingleton:
    """Singleton for table formatter"""

    _instance = None
    formatter = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            logger.info("Initializing AutoTableFormatter...")
            cls._instance = super(TableFormatterSingleton, cls).__new__(cls)
            cls._instance.formatter = AutoTableFormatter() 
        return cls._instance

class GMFTParsedTable(GenericParsedTable):
    def __init__(self, table: CroppedTable, page_num: int) -> None:
        super().__init__(
            page_number=page_num, bbox=table.bbox
        )  # Initialize the field from the abstract class
        self._table = table
        self.failed = False
        self.formatter = TableFormatterSingleton().formatter

        # Formatter is passed externally
        # self.formatter = formatter

    @cached_property
    def _captions(self) -> List[str]:
        # return ""
        return [c for c in self._table.captions() if c.strip()]

    @cached_property
    def caption(self) -> str:
        return "\n".join(set(self._captions))

    @property
    def df(self) -> Optional[pd.DataFrame]:
        ft = self.formatter.extract(self._table)
        try:
            df = ft.df()
        except ValueError as ex:
            logger.error(f"Couldn't extract df on page {self.page_num}: {str(ex)}")
            self.failed = True
            return None

            # config = AutoFormatConfig()
            # config.total_overlap_reject_threshold = 0.8
            # config.large_table_threshold = 0

            # try:
            # logger.info("\tTrying to reover")
            # df = ft.df(config_overrides = config)
            # except ValueError:
            # logger.error(f"\tCouldn't recover, page {self.page_num}: {str(ex)}")
            # return None

        return df

    @property
    def xml(self) -> List[str]:
        if self.df is None:
            return list()
        return pandas_df_to_xml(self.df)


@dataclass
class PageTables:
    page_num: int
    cropped_tables: List[CroppedTable]

    @property
    def n_tables(self):
        return len(self.cropped_tables)


class GMFTParser:
    def __init__(self, fn: Path) -> None:
        self.fn = fn
        self._doc = None
        self._parsed_tables = None
        
        # logger.info("Initializing Table Formatter.")
        # self.formatter = AutoTableFormatter()

    def detect_page_tables(self) -> Tuple[List[PageTables], Any]:
        """Detects tables in a document and returns list of page tables"""

        logger.info("Detecting tables...")
        doc = PyPDFium2Document(self.fn)
        detector = TableDetector()
        pt = []

        for page in doc:
            pt.append(
                PageTables(
                    page_num=page.page_number, cropped_tables=detector.extract(page)
                )
            )

        return pt, doc

    @property
    def parsed_tables(self) -> List[GenericParsedTable]:
        if self._parsed_tables is None:
            page_tables, self._doc = self.detect_page_tables()
            logger.info("Parsing tables ...")

            out_tables = []

            for page_table in page_tables:
                for cropped_table in page_table.cropped_tables:
                    out_tables.append(
                        GMFTParsedTable(cropped_table, page_table.page_num)
                    )
            self._parsed_tables = out_tables
        return self._parsed_tables


if __name__ == "__main__":
    # fn = Path("/home/snexus/Downloads/ws90.pdf")
    # fn = Path("/home/snexus/Downloads/SSRN-id2741701.pdf")
    fn = Path("/home/snexus/Downloads/Table_Example1.pdf")

    parser = GMFTParser(fn=fn)
    for p in parser.parsed_tables:
        print("-------------")
        print(p.page_num)
        print(p.caption)
        print(p.bbox)
        print('\n'.join(p.xml))
    
    # chunks = pdf_table_splitter(parsed_table=parser.parsed_tables[7], max_size = 1024)
    # for chunk in chunks:
        # print("\n=========== CHUNK START =============\n")
        # print(chunk['text'])
    # # print(chunks)
    del parser

