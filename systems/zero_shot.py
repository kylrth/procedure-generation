import logging

from .interface import SystemInterface
from .model import Model, log


class ZeroShot(SystemInterface):
    """This model prompts an LM to generate text zero-shot, meaning no examples are provided.

    The template should expect a "title" input variable.
    """

    instructions: str = (
        "Please generate a recipe from the given title. Provide a list of ingredients and a list "
        "of instructions."
    )

    def __init__(self, model: Model):
        self.model = model

    def generate(self, title: str, logger: logging.Logger | None = None) -> list[str]:
        prompt = self.model.build_prompt(title, self.instructions)
        completion = self.model.generate(prompt)

        log(logger, "ZeroShot", prompt, completion)

        return completion

    async def agenerate(self, title: str, logger: logging.Logger | None = None) -> list[str]:
        prompt = self.model.build_prompt(title, self.instructions)
        completion = await self.model.agenerate(prompt)

        log(logger, "ZeroShot", prompt, completion)

        return completion
