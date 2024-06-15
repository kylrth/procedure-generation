from typing import ClassVar

from langchain_core.messages import BaseMessage

from dataset import Doc
from utils import log

from .interface import Response, System
from .model import Model


class FewShot(System):
    """This model prompts an LM to generate a response with a fixed set of examples."""

    model: Model
    dataset: str

    def __init__(self, model: Model, dataset: str, shots: list[Doc] | None = None):
        self.model = model
        self.dataset = dataset
        self.shots = self.build_context(shots if shots is not None else [])

    _example_name: ClassVar[dict[str, str]] = {
        "lcstep": "DOCUMENTATION",
        "recipenlg": "RECIPE",
        "champ": "EXAMPLE",
    }

    def build_context(self, docs: list[Doc]) -> str:
        out = ""
        for doc in docs:
            out += f"\n\n{self._example_name[self.dataset]} '{doc.title}':\n\n{doc.contents}"

        return out[2:]  # skip first "\n\n"

    async def generate(self, logger: log.InstanceLogger, query: str, input_: str) -> Response:
        prompt = await self.build_prompt(logger, query, input_)
        completion = await self.model.generate(prompt)

        return self._make_result(prompt, completion)

    _instructions: ClassVar[dict[str, str]] = {
        "lcstep": (
            "Please generate high-level steps to accomplish the specified goal using the LangChain "
            "Python library. Don't include code, extraneous commentary, or examples, but do refer "
            "to the specific LangChain APIs (or other APIs) used in each step. Don't produce any "
            "text other than the list of steps. Use any of the provided reference documentation to "
            "answer the question."
        ),
        "recipenlg": (
            "Please generate high-level steps to create a recipe for the specified food. Don't "
            "include extraneous commentary, or examples, but do refer to the special "
            "characteristics and state of the ingredients used in each step. Don't produce any "
            "text other than the list of steps. Use any of the provided reference recipes to "
            "answer the question."
        ),
        "champ": (
            "Please generate high-level steps to solve the given math problem. Don't include code, "
            "extraneous commentary, or examples, but do refer to the concepts and hints used in "
            "each step. Don't produce any text other than the list of steps. Use any of the "
            "provided similar problems and solutions to answer the question."
        ),
    }

    _prompt_inst: ClassVar[dict[str, str]] = {
        "lcstep": (
            "Please generate a list of instructions to accomplish '{query}' using the procedures "
            "above. Create and use these resources in your response: {input_}."
        ),
        "recipenlg": (
            "Please generate a list of instructions to accomplish '{query}' using the recipes "
            "above. Use these ingredients in your response: {input_}."
        ),
        "champ": (
            "Please generate a list of instructions to solve '{query}' using the examples above. "
            "Use this additional information in preparing your response: {input_}."
        ),
    }

    # This function is async so that RAG can inherit and override it
    async def build_prompt(
        self, logger: log.InstanceLogger, query: str, input_: str
    ) -> str | list[BaseMessage]:
        context = self.shots
        msg_prompt = (
            context + "\n\n" + self._prompt_inst[self.dataset].format(query=query, input_=input_)
        )

        out = self.model.build_prompt(msg_prompt, self._instructions[self.dataset])

        logger.write(f"prompt to model {self.model.name}:\n")
        logger.log_prompt(out)

        return out

    def _make_result(self, prompt: str | list[BaseMessage], completion: str):
        return Response(
            answer=self.parse_completion(completion),
            model=self.model.name,
            input_tokens=self.model.get_num_tokens(prompt),
            output_tokens=self.model.get_num_tokens(completion),
        )
