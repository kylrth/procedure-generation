import logging

from .interface import System
from .model import Model, log


class ZeroShot(System):
    """This model prompts an LM to generate text zero-shot, meaning no examples are provided."""

    instructions: str = (
        "Please generate ordered steps to accomplish the specified goal using the LangChain Python "
        "library."
    )

    def __init__(self, model: Model):
        self.model = model

    def generate(self, goal: str, logger: logging.Logger | None = None) -> list[str]:
        prompt = self.model.build_prompt(goal, self.instructions)
        completion = self.model.generate(prompt)

        log(logger, "ZeroShot", prompt, completion)

        return completion

    async def agenerate(self, goal: str, logger: logging.Logger | None = None) -> list[str]:
        prompt = self.model.build_prompt(goal, self.instructions)
        completion = await self.model.agenerate(prompt)

        log(logger, "ZeroShot", prompt, completion)

        return completion
