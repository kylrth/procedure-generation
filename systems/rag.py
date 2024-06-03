import textwrap
from typing import ClassVar

from langchain.schema import BaseMessage

import retrieval
from dataset import Doc
from utils import log

from .few_shot import FewShot
from .model import Model


class RAG(FewShot):
    """This model prompts an LM to generate the response few-shot, but with the examples provided by
    searching a vector store for texts with similar embeddings to the query."""

    model: Model
    docs: retrieval.Doc_store
    k: int

    def __init__(self, model: Model, docs: retrieval.Doc_store, k: int, dataset: str):
        super().__init__(model, dataset)
        self.docs = docs
        self.k = k

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

    def build_prompt(
        self, logger: log.InstanceLogger, query: str, input_: str
    ) -> str | list[BaseMessage]:
        docs = self.docs.get_docs(f"{query} using {input_}")
        logger.write(f"retrieved {len(docs)} docs\n")

        context = self.build_context(docs)
        msg_prompt = (
            context + "\n\n" + self._prompt_inst[self.dataset].format(query=query, input_=input_)
        )

        out = self.model.build_prompt(msg_prompt, self._instructions[self.dataset])

        logger.write(f"prompt to model {self.model.name}:\n")
        logger.write(textwrap.indent(log.messages_to_string(out), "  ") + "\n")

        return out

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
