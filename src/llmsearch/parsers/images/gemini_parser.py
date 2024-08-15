"""Experimental Image Analyzer"""

import os
from pathlib import Path

import google.generativeai as genai
import PIL.Image
from loguru import logger

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
