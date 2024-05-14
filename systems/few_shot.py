from langchain.schema import BaseMessage

from .interface import Response, System
from .model import Model


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

    def generate(self, query: str) -> Response:
        prompt = self.model.build_prompt(query, self.instructions, self.shots)
        completion = self.model.generate(prompt)

        return self._make_result(query, prompt, completion)

    async def agenerate(self, query: str) -> Response:
        prompt = self.model.build_prompt(query, self.instructions, self.shots)
        completion = await self.model.agenerate(prompt)

        return self._make_result(query, prompt, completion)

    def _make_result(self, query: str, prompt: str | list[BaseMessage], completions: list[str]):
        return Response(
            query,
            prompt,
            completions[0],
            self.model.model,
            [],
            prompt if isinstance(prompt, str) else "\n\n".join(str(msg.content) for msg in prompt),
        )
