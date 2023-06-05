import logging
from typing import List, Optional, Tuple

from langchain.prompts.example_selector.base import BaseExampleSelector

from .interface import SystemInterface
from .model import Model, log


class FewShot(SystemInterface):
    """This model prompts an LM to generate text few-shot, with the examples provided by searching a
    vector store for texts with similar embeddings to the title."""

    def __init__(self, model: Model, instructions: str, selector: BaseExampleSelector):
        self.model = model
        self.instructions = instructions
        self.selector = selector

    def generate(self, title: str, logger: Optional[logging.Logger] = None) -> List[str]:
        prompt = self.model.build_prompt(title, self.instructions, self.get_examples(title))
        completion = self.model.generate(prompt)

        log(logger, "FewShot", prompt, completion)

        return completion

    async def agenerate(self, title: str, logger: Optional[logging.Logger] = None) -> List[str]:
        prompt = self.model.build_prompt(title, self.instructions, self.get_examples(title))
        completion = await self.model.agenerate(prompt)

        log(logger, "FewShot", prompt, completion)

        return completion

    def get_examples(self, title: str) -> List[Tuple[str, str]]:
        """Returns the examples that will be inserted into the prompt."""
        examples = self.selector.select_examples({"title": title})

        return [(example["title"], example["recipe"]) for example in examples]
