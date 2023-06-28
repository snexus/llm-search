from enum import Enum, auto
from pathlib import Path
from typing import List, Union

from loguru import logger
from unstructured.documents.elements import NarrativeText, Text, Title
from unstructured.partition.html import partition_html
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.epub import partition_epub


if __name__ == "__main__":    
    n_output_elements = 2000
    path = "/storage/llm/microsoft-docs/dbks.pdf"
    
    elements = partition_pdf(filename = path, strategy="fast", include_page_breaks=True, infer_table_structure = True)
    print(elements)
    # print("Done processing. Writing output...")
    
    # with open("output.txt", mode="w") as f:
    #     for el in elements[:n_output_elements]:
    #         s = f"<{el.category} - {el.metadata.page_number}> {el.text}\n"
    #         f.write(s)
        
    
