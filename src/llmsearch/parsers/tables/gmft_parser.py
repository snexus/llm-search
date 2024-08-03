from functools import cached_property
import pandas as pd
from typing import Any, List, Optional, Tuple
from gmft.pdf_bindings import PyPDFium2Document
from gmft import CroppedTable, TableDetector, AutoFormatConfig, AutoTableFormatter, TATRTableFormatter
from pathlib import Path
from loguru import logger
from dataclasses import dataclass
# from pydantic import BaseModel, Field


class ParsedTable:
    def __init__(self, table: CroppedTable, page_num: int) -> None:
        self._table = table
        self.page_num = page_num
        self.config = AutoFormatConfig()
        self.formatter = AutoTableFormatter()

    @cached_property
    def captions(self) -> List[str]:
        try:
            return self._table.captions()
        except Exception as ex:
            logger.error(f"Couldn't parse captions: {str(ex)}")
            return []
    
    @property
    def df(self) -> Optional[pd.DataFrame]:
        ft =  self.formatter.extract(self._table)
        try:
            df = ft.df()
        except ValueError as ex:
            logger.error(f"Couldn't extract df on page {self.page_num}: {str(ex)}")

            config = AutoFormatConfig()
            config.total_overlap_reject_threshold = 0.8
            config.large_table_threshold = 0

            try:
                logger.info("\tTrying to reover")
                df = ft.df(config_overrides = config)
            except ValueError:
                logger.error(f"\tCouldn't recover, page {self.page_num}: {str(ex)}")
                return None

        return df
    
    @property
    def xml(self) -> str:
        def func(row):
            xml = ['<row>']
            for field in row.index:
                xml.append('  <col name="{0}">{1}</col>'.format(field, row[field]))
            xml.append('</row>')
            return '\n'.join(xml)
        df = self.df
        if df is None:
            return ""
        items = df.apply(func, axis=1)
        return '\n'.join(items)


@dataclass
class PageTables:
    page_num: int
    cropped_tables: List[CroppedTable]

    @property
    def n_tables(self):
        return len(self.cropped_tables)


class GMFTTableParser:
    def __init__(self, fn: Path) -> None:
        self.fn = fn
        logger.info("Loading document.")

    def detect_tables(self) -> Tuple[List[PageTables], Any]:
        """Detects tables in a document and returns list of page tables"""

        logger.info("Detecting tables...")
        doc = PyPDFium2Document(fn)
        detector = TableDetector()
        pt = []

        for page in doc:
            pt.append(
                PageTables(page_num=page.page_number, cropped_tables=detector.extract(page))
            )

        # doc.close()
        return pt, doc


if __name__ == "__main__":
    # fn = Path("/home/snexus/Downloads/ws90.pdf")
    fn = Path("/home/snexus/Downloads/SSRN-id2741701.pdf")

    parser = GMFTTableParser(fn=fn)
    page_tables, doc = parser.detect_tables()

    for t in page_tables:
        print(f" ========= PAGE {t.page_num} ============")
        print(f"Number detected tables: {t.n_tables}")
        for table in t.cropped_tables:
            p = ParsedTable(table, t.page_num)
            print(p.captions)
            print(p.xml)
            print('-------------')

    doc.close()

