import importlib
import io
from collections import defaultdict
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import PIL.Image
import pymupdf
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential

from llmsearch.config import PDFImageParser, PDFImageParseSettings
from llmsearch.parsers.markdown import markdown_splitter

# Define a mapping of PDFImageParser to corresponding analyzer classes and config
ANALYZER_MAPPING: Dict[PDFImageParser, Any] = {
    PDFImageParser.GEMINI_15_FLASH: {
        "import_path": "llmsearch.parsers.images.gemini_parser",  # Import path for lazy loading
        "class_name": "GeminiImageAnalyzer",
        "params": {"model_name": "gemini-1.5-flash"},
    },

    PDFImageParser.GEMINI_15_PRO: {
        "import_path": "llmsearch.parsers.images.gemini_parser",  # Import path for lazy loading
        "class_name": "GeminiImageAnalyzer",
        "params": {"model_name": "gemini-1.5-pro"},
    },
}


def create_analyzer(image_analyzer: PDFImageParser, **additional_params):
    analyzer_info = ANALYZER_MAPPING.get(image_analyzer)

    if analyzer_info is None:
        raise ValueError(f"Unsupported image analyzer type: {image_analyzer}")

    # Lazy load the module
    module = importlib.import_module(analyzer_info["import_path"])
    analyzer_class = getattr(module, analyzer_info["class_name"])
    analyzer_params = analyzer_info["params"]

    params = {**analyzer_params, **additional_params}

    return analyzer_class(**params)


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
        image_analyzer: Callable,
        save_output: bool = True,
        max_base_width: int = 1280,
        min_width: int = 640,
        min_height: int = 200,
    ):
        self.pdf_fn = pdf_fn
        self.temp_folder = temp_folder
        self.image_analyzer = image_analyzer
        self.save_output = save_output
        self.max_base_width = max_base_width
        self.min_width = min_width
        self.min_height = min_height

    def prepare_and_clean_folder(self):
        if not self.temp_folder.exists():
            self.temp_folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created folder: {self.temp_folder}")
        else:
            for file in self.temp_folder.iterdir():
                if file.is_file():
                    file.unlink()
                    logger.debug(f"Deleted file: {file}")

    def extract_images(self) -> List[PDFImage]:
        self.prepare_and_clean_folder()
        doc = pymupdf.open(self.pdf_fn)
        out_images = []

        for page in doc:
            for img in page.get_images():
                xref = img[0]
                data = doc.extract_image(xref)
                try:
                    out_fn = self._resize_and_save_image(data, page.number, xref)
                except Exception as ex:
                    logger.error(f"An exception occured when opening the image: {str(ex)}")
                    out_fn = None
                if out_fn:
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
        image_data = data.get("image")
        if not image_data:
            return

        with PIL.Image.open(io.BytesIO(image_data)) as img:
            if img.size[1] < self.min_height or img.size[0] < self.min_width:
                logger.debug(
                    f"Image on page {page_num}, xref {xref_num} is too small. Skipping extraction..."
                )
                return None

            wpercent = self.max_base_width / float(img.size[0])
            if wpercent < 1:
                hsize = int(float(img.size[1]) * wpercent)
                img = img.resize(
                    (self.max_base_width, hsize), PIL.Image.Resampling.LANCZOS
                )

            out_fn = (
                self.temp_folder
                / f"{self.pdf_fn.stem}_page_{page_num}_xref_{xref_num}.png"
            )
            logger.debug(f"Saving file: {out_fn}")
            img.convert("RGB").save(out_fn)

        return out_fn

    def analyze_images_threaded(
        self, extracted_images: List[PDFImage], max_threads: int = 10
    ):
        with ThreadPool(max_threads) as pool:
            results = pool.starmap(
                analyze_single_image,
                [
                    (img, self.image_analyzer, i)
                    for i, img in enumerate(extracted_images)
                ],
            )

        if self.save_output:
            for result in results:
                with open(str(result.image_fn).replace(".png", ".md"), "w") as file:
                    file.write(result.markdown)

        return results


def log_attempt_number(retry_state):
    error_message = str(retry_state.outcome.exception())
    logger.error(
            f"API call attempt {retry_state.attempt_number} failed with error: {error_message}. Retrying..."
        )


def on_retry_failed(retry_state):
    logger.error("API calls failed for maximum number of retries. Skipping image processing for this graph.")
    return retry_state.args[0]

@retry(
    wait=wait_random_exponential(min=5, max=60),
    stop=stop_after_attempt(6),
    after=log_attempt_number,
    retry_error_callback= on_retry_failed,
)
def analyze_single_image(
    pdf_image: PDFImage, image_analyzer: Callable, i: int
) -> PDFImage:
    pdf_image.markdown = image_analyzer.analyze(pdf_image.image_fn)
    return pdf_image


def get_image_chunks(
    path: Path,
    max_size: int,
    image_parse_setting: PDFImageParseSettings,
    cache_folder: Path,
) -> Tuple[List[dict], Dict[int, List[Tuple[float]]]]:

    analyzer = create_analyzer(
        image_parse_setting.image_parser,
        system_instruction=image_parse_setting.system_instruction,
        user_instruction=image_parse_setting.user_instruction,
    )
    image_parser = GenericPDFImageParser(
        pdf_fn=path,
        temp_folder=cache_folder / "pdf_images_temp",
        image_analyzer=analyzer,
    )
    extracted_images = image_parser.extract_images()
    parsed_images = image_parser.analyze_images_threaded(extracted_images)

    out_blocks = []
    img_bboxes = defaultdict(list)

    for img in parsed_images:
        out_blocks += markdown_splitter(
            path=str(img.image_fn).replace(".png", ".md"), max_chunk_size=max_size
        )
        img_bboxes[img.page_num].append(img.bbox)

    return out_blocks, img_bboxes


if __name__ == "__main__":

    res = get_image_chunks(
        path=Path("/home/snexus/Downloads/Graph_Example2.pdf"),
        max_size=1024,
        image_parse_setting=PDFImageParseSettings(image_parser= PDFImageParser.GEMINI_15_PRO),
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
