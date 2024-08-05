from typing import List
import pandas as pd
from loguru import logger

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
        logger.warning("Caption is too large compared to max char size, trimming down...")
        caption = caption[:int(max_size / max_caption_size_ratio)]
    
    header = f"```xml:\n"
    if include_caption and caption:
        header = header + f"<caption>{caption}</caption>\n"

    footer = f"```"

    current_text = header
    for el in xml_elements:

        # If new element is too big, trim it (shouldn't happen) 
        if len(el) > max_size:
            logger.warning("xml element is larger than allowed max char size. Flushing..")
            # el = el[:max_size-len(header)-3]
            all_chunks.append({"text": current_text+el+footer, "metadata": metadata})
            current_text = header
        
        # if current text is already large and doesn't fit the new element, flush it
        elif len(current_text + el) >= max_size:
            all_chunks.append({"text": current_text + footer, "metadata": metadata})
            current_text = header + el + "\n"
        else:
            current_text += el +"\n"

    # Flush the last chunk
    all_chunks.append({"text": current_text + footer, "metadata": metadata})
    return all_chunks
        


    

