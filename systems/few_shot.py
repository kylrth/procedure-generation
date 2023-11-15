import logging

from .interface import System
from .model import Model, log


class FewShot(System):
    """This model prompts an LM to generate text with a fixed set of examples."""

    instructions: str = (
        "Please generate high-level steps to accomplish the specified goal using the LangChain "
        "Python library. Don't include code, extraneous commentary, or examples, but do refer to "
        "the specific LangChain APIs (or other APIs) used in each step. Don't produce any text "
        "other than the list of steps."
    )

    def __init__(self, model: Model, shots: list[tuple[str, str]] | None = None):
        self.model = model
        self.shots = shots if shots is not None else []

    def generate(self, goal: str, logger: logging.Logger | None = None) -> list[str]:
        prompt = self.model.build_prompt(goal, self.instructions, self.shots)
        completion = self.model.generate(prompt)

        log(logger, f"{len(self.shots)}Shot", prompt, completion)

        return completion

    async def agenerate(self, goal: str, logger: logging.Logger | None = None) -> list[str]:
        prompt = self.model.build_prompt(goal, self.instructions, self.shots)
        completion = await self.model.agenerate(prompt)

        log(logger, f"{len(self.shots)}Shot", prompt, completion)

        return completion
