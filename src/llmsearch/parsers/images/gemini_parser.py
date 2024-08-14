"""Experimental Image Analyzer"""

from typing import List, Optional
import pymupdf
import PIL.Image
import io
from pathlib import Path
from loguru import logger

from pathlib import Path
from llmsearch.parsers.images.generic import PDFImage
import google.generativeai as genai
import os
from multiprocessing.pool import ThreadPool

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

api_key = os.environ.get("GOOGLE_API_KEY", "")
if not api_key:
    logger.error("Please specify GOOGLE_API_KEY=xxx in .env file.")
    raise ValueError

genai.configure(api_key=api_key)


class GeminiImageAnalyzer:
    def __init__(
        self,
        model_name: str,
        instruction: str = """From the image, extract detailed quantitative and qualitative data points.""",
    ):
        self.model_name = model_name
        self.instruction = instruction
        self.model = genai.GenerativeModel(
            model_name,
            system_instruction="""You are an research assistant. You analyze the image to extract detailed information. Response must be a Markdown string in the follwing format:

- First line is a heading with image caption, starting with '# '
- Second line is empty
- From the third line on - detailed data points and related metadata, extracted from the image, in Markdown format. Don't use Markdown tables. 
""",
            generation_config=genai.types.GenerationConfig(
                # Only one candidate for now.
                candidate_count=1,
                temperature=0.2,  # Reduce creativity for graph analysis.
            ),
        )
        logger.info(f"Initialized `{model_name}` model.")

    def analyze(self, image_fn) -> str:
        logger.info(f"\tAnalyzing image: {image_fn}")
        # return f"THIS IS MD FROM {image_fn}"
        image = PIL.Image.open(image_fn)
        response = self.model.generate_content(
            [
                self.instruction,
                image,
            ],
            stream=False,
        )
        response.resolve()
        return response.text


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

            out_fn = self.temp_folder / f"page_{page_num}_xref_{xref_num}.png"
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
                with open(str(r.image_fn)[:-3] + ".json", "w") as file:
                    file.write(r.model_dump_json(indent=4))

        return results


def log_attempt_number(retry_state):
    """return the result of the last call attempt"""
    logger.error(f"Retrying: {retry_state.attempt_number}...")


@retry(
    wait=wait_random_exponential(min=5, max=60),
    stop=stop_after_attempt(6),
    after=log_attempt_number,
)
def analyze_single_image(pdf_image: PDFImage, image_analyzer, i: int) -> PDFImage:
    fn = pdf_image.image_fn
    pdf_image.markdown = image_analyzer.analyze(fn)
    return pdf_image


if __name__ == "__main__":
    image_parser = GenericPDFImageParser(
        pdf_fn=Path("/home/snexus/Downloads/Graph_Example1.pdf"),
        temp_folder=Path("./output_images"),
        image_analyzer=GeminiImageAnalyzer(model_name="gemini-1.5-flash"),
        # image_analyzer=GeminiImageAnalyzer(model_name="gemini-1.5-pro-exp-0801")
    )

    all_images = image_parser.extract_images()
    final_images = image_parser.analyze_images_threaded(all_images)
    print(final_images)
    logger.info("DOne.")

    # analyzer = GeminiImageAnalyzer(model_name="gemini-1.5-flash")
    # # analyzer = GeminiImageAnalyzer(model_name="gemini-1.5-pro-exp-0801")
    # out = analyzer.analyze(image_fn=Path("./output_images/page_6_xref_301.png"))

    # print(out)
