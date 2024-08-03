import pandas as pd

from abc import ABC, abstractmethod


class GenericParsedTable(ABC):
    def __init__(self, page_number: int):
        self.page_num = page_number  # Common field

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
    def xml(self) -> str:
        """Returns xml representation of the table"""
        pass


def pandas_df_to_xml(df: pd.DataFrame) -> str:
    """Converts Pandas df to a simplified xml representation digestible by LLMs

    Args:
        df (pd.DataFrame): Pandas df

    Returns:
        str: xml string
    """

    def func(row):
        xml = ["<row>"]
        for field in row.index:
            xml.append('  <col name="{0}">{1}</col>'.format(field, row[field]))
        xml.append("</row>")
        return "\n".join(xml)

    items = df.apply(func, axis=1)
    return "\n".join(items)
