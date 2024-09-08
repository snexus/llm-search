from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.prompts import PromptTemplate


class AbstractLLMModel(ABC):
    def __init__(self, prompt_template: Optional[str] = None) -> None:
        self.prompt_template = prompt_template

    @property
    @abstractmethod
    def model(self):
        raise NotImplementedError

    @property
    def prompt(self) -> Optional[PromptTemplate]:
        if self.prompt_template:
            return PromptTemplate(
                input_variables=["context", "question"], template=self.prompt_template
            )
        return None
