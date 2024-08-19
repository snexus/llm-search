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
        system_instruction: str,
        user_instruction: str
    ):
        self.model_name = model_name
        self.instruction = user_instruction
        print(system_instruction, user_instruction)
        self.model = genai.GenerativeModel(
            model_name,
            system_instruction = system_instruction,
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
