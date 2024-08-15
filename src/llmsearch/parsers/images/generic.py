from collections import defaultdict
import io
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import PIL.Image
import pymupdf
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential

from llmsearch.config import PDFImageParser
from llmsearch.parsers.markdown import markdown_splitter


class PDFImage(BaseModel):
    image_fn: Path
    page_num: int
    bbox: Tuple[float, float, float, float]
    markdown: str = ""


class GenericPDFImageParser:
    def __init__(
        self,
        pdf_fn: Path,
        temp_folder: Path,
        image_analyzer,
        save_output=True,
        max_base_width: int = 1280,
        min_width: int = 640,
        min_height: int = 200,
    ):
        self.pdf_fn = pdf_fn
        self.max_base_width = max_base_width
        self.temp_folder = temp_folder
        self.min_width = min_width
        self.min_height = min_height
        self.image_analyzer = image_analyzer
        self.save_output = save_output

    def prepare_and_clean_folder(self):
        # Check if the folder exists
        if not self.temp_folder.exists():
            # Create the folder if it doesn't exist
            self.temp_folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created folder: {self.temp_folder}")
        else:
            for file in self.temp_folder.iterdir():
                if file.is_file():
                    file.unlink()  # Delete the file
                    logger.info(f"Deleted file: {file}")

    def extract_images(self) -> List[PDFImage]:
        self.prepare_and_clean_folder()

        doc = pymupdf.open(self.pdf_fn)
        out_images = []

        for page in doc:
            page_images = page.get_images()
            for img in page_images:
                xref = img[0]
                data = doc.extract_image(xref=xref)
                out_fn = self._resize_and_save_image(
                    data=data,
                    page_num=page.number,
                    xref_num=xref,
                )
                if out_fn is not None:
                    out_images.append(
                        PDFImage(
                            image_fn=out_fn,
                            page_num=page.number,
                            bbox=(img[1], img[2], img[3], img[4]),
                        )
                    )

        return out_images

    def _resize_and_save_image(
        self,
        data: dict,
        page_num: int,
        xref_num: int,
    ) -> Optional[Path]:
        
        image = data.get("image", None)
        if image is None:
            return

        with PIL.Image.open(io.BytesIO(image)) as img:
            if img.size[1] < self.min_height or img.size[0] < self.min_width:
                logger.info(
                    f"Image on page {page_num}, xref {xref_num} is too small. Skipping extraction..."
                )
                return None
            wpercent = self.max_base_width / float(img.size[0])

            # Resize the image, if needed
            if wpercent < 1:
                hsize = int((float(img.size[1]) * float(wpercent)))
                img = img.resize(
                    (self.max_base_width, hsize), PIL.Image.Resampling.LANCZOS
                )

            out_fn = self.temp_folder / (str(self.pdf_fn.stem) + f"_page_{page_num}_xref_{xref_num}.png")
            logger.info(f"Saving file: {out_fn}")
            img.save(out_fn, mode="wb")
        return Path(out_fn)

    def analyze_images_threaded(
        self, extracted_images: List[PDFImage], max_threads: int = 10
    ):
        with ThreadPool(max_threads) as pool:
            results = pool.starmap(
                analyze_single_image,
                [
                    (pdf_image, self.image_analyzer, i)
                    for i, pdf_image in enumerate(extracted_images)
                ],
            )

        if self.save_output:
            for r in results:
                with open(str(r.image_fn)[:-3] + ".md", "w") as file:
                    file.write(r.markdown)

        return results


def log_attempt_number(retry_state):
    """return the result of the last call attempt"""
    logger.error(f"API call attempt failed. Retrying: {retry_state.attempt_number}...")


@retry(
    wait=wait_random_exponential(min=5, max=60),
    stop=stop_after_attempt(6),
    after=log_attempt_number,
)
def analyze_single_image(pdf_image: PDFImage, image_analyzer, i: int) -> PDFImage:
    fn = pdf_image.image_fn
    pdf_image.markdown = image_analyzer.analyze(fn)
    return pdf_image


def get_image_chunks(
    path: Path, max_size: int, image_analyzer: PDFImageParser, cache_folder: Path
) -> Tuple[List[dict], Dict[int, List[Tuple[float]]]]:
    if image_analyzer is PDFImageParser.GEMINI_15_FLASH:
        from llmsearch.parsers.images.gemini_parser import GeminiImageAnalyzer
        analyzer = GeminiImageAnalyzer(model_name="gemini-1.5-flash")

    image_parser = GenericPDFImageParser(
        pdf_fn=path,
        temp_folder=cache_folder / "pdf_images_temp",
        image_analyzer=analyzer,
        # image_analyzer=GeminiImageAnalyzer(model_name="gemini-1.5-pro-exp-0801")
    )

    extracted_images = image_parser.extract_images()
    parsed_images = image_parser.analyze_images_threaded(extracted_images)

    out_blocks = []
    img_bboxes = defaultdict(list)

    for img in parsed_images:
        print(str(img.image_fn) + ".md")
        out_blocks += markdown_splitter(path=str(img.image_fn)[:-3] + ".md", max_chunk_size=max_size)
        img_bboxes[img.page_num].append(img.bbox)

    return out_blocks, img_bboxes


if __name__ == "__main__":

    res = get_image_chunks(
        path=Path("/home/snexus/Downloads/Graph_Example2.pdf"),
        max_size=1024,
        image_analyzer=PDFImageParser.GEMINI_15_FLASH,
        cache_folder=Path("./output_images"),
    )

    print(res)
    # from llmsearch.parsers.images.gemini_parser import GeminiImageAnalyzer

    # image_parser = GenericPDFImageParser(
    # pdf_fn=Path("/home/snexus/Downloads/Table_Example5.pdf"),
    # temp_folder=Path("./output_images"),
    # image_analyzer=GeminiImageAnalyzer(model_name="gemini-1.5-flash"),
    # # image_analyzer=GeminiImageAnalyzer(model_name="gemini-1.5-pro-exp-0801")
    # )

    # all_images = image_parser.extract_images()
    # final_images = image_parser.analyze_images_threaded(all_images)
    # print(final_images)
    # logger.info("DOne.")

    # analyzer = GeminiImageAnalyzer(model_name="gemini-1.5-flash")
    # # analyzer = GeminiImageAnalyzer(model_name="gemini-1.5-pro-exp-0801")
    # out = analyzer.analyze(image_fn=Path("./output_images/page_6_xref_301.png"))

    # print(out)
