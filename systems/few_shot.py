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

    def generate(self, query: str, _input: str) -> Response:
        prompt = self._make_prompt(query, _input)
        completion = self.model.generate(prompt)

        return self._make_result(prompt, completion)

    async def agenerate(self, query: str, _input: str) -> Response:
        prompt = self._make_prompt(query, _input)
        completion = await self.model.agenerate(prompt)

        return self._make_result(prompt, completion)

    def _make_prompt(self, query: str, _input: str) -> str | list[BaseMessage]:
        # merge the input string into the query
        query = f"{query} using {_input}"

        return self.model.build_prompt(query, self.instructions, self.shots)

    def _make_result(self, prompt: str | list[BaseMessage], completions: list[str]):
        return Response(
            self.parse_completion(completions[0]),
            prompt,
            self.model.model,
        )
