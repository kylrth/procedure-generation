from typing import List

from .interface import SystemInterface
from .model import Model


class ZeroShot(SystemInterface):
    """This model prompts an LM to generate text zero-shot, meaning no examples are provided.

    The template should expect a "title" input variable.
    """

    def __init__(self, model: Model, instructions: str):
        self.model = model
        self.instructions = instructions

    def generate(self, title: str) -> List[str]:
        return self.model(title, self.instructions)
