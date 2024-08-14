from pathlib import Path
from typing import List, Tuple
from loguru import logger
from pydantic import BaseModel

from abc import ABC, abstractmethod


class PDFImage(BaseModel):
    image_fn: Path
    page_num: int
    bbox: Tuple[float, float, float, float]
    markdown: str = ""
