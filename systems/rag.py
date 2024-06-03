import logging
import textwrap
from typing import ClassVar

import weaviate
import weaviate.classes.config as wc
from langchain.schema import BaseMessage

import retrieval
from dataset import Doc
from utils import log

from .few_shot import FewShot
from .model import Model


def setup_store(
    logger: logging.Logger, store: weaviate.WeaviateClient, name: str, desc: str
) -> weaviate.collections.Collection:
    """Create a generic vector store for RAG."""
    if store.collections.exists(name):
        logger.info("reusing existing Weaviate collection")
        out = store.collections.get(name)
    else:
        logger.info("creating new Weaviate collection")
        out = store.collections.create(
            name=name,
            description=desc,
            properties=[
                wc.Property(
                    name="title",
                    data_type=wc.DataType.TEXT,
                    description="The title of the document",
                ),
                wc.Property(
                    name="chunk",
                    data_type=wc.DataType.INT,
                    description="Zero-indexed chunk number in the document",
                    skip_vectorization=True,
                    vectorize_property_name=False,
                ),
                wc.Property(
                    name="contents",
                    data_type=wc.DataType.TEXT,
                    description="The contents of (this chunk of) the document",
                ),
            ],
            vectorizer_config=wc.Configure.Vectorizer.none(),
        )
    return out


class RAG(FewShot):
    """This model prompts an LM to generate the response few-shot, but with the examples provided by
    searching a vector store for texts with similar embeddings to the query."""

    docs: weaviate.collections.Collection
    k: int

    def __init__(self, model: Model, docs: weaviate.collections.Collection, k: int, dataset: str):
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
        docs = self.get_docs(f"{query} using {input_}")
        logger.write(f"retrieved {len(docs)} docs\n")

        context = self.build_context(docs)
        msg_prompt = (
            context + "\n\n" + self._prompt_inst[self.dataset].format(query=query, input_=input_)
        )

        out = self.model.build_prompt(msg_prompt, self._instructions[self.dataset])

        logger.write(f"prompt to model {self.model.name}:\n")
        logger.write(textwrap.indent(log.messages_to_string(out), "  ") + "\n")

        return out

    def get_docs(self, query: str) -> list[Doc]:
        """Returns the docs that will be inserted into the prompt."""
        embedded_query = retrieval.get_embeds([query])[0]

        res = self.docs.query.near_vector(
            near_vector=embedded_query.tolist(),
            limit=self.k,
            return_properties=["title", "contents"],
        )

        out = []
        for obj in res.objects:
            out.append(Doc(obj.properties["title"], obj.properties["contents"]))

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
