import logging

import weaviate
import weaviate.classes.config as wc
from langchain.schema import BaseMessage

import retrieval
from dataset import Doc

from .interface import Response, System
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


rag_task: dict[str, str] = {
    "lcstep": (
        "Please generate high-level steps to accomplish the specified goal using the LangChain "
        "Python library. Don't include code, extraneous commentary, or examples, but do refer to "
        "the specific LangChain APIs (or other APIs) used in each step. Don't produce any text "
        "other than the list of steps. Use any of the provided reference documentation to answer "
        "the question."
    ),
    "recipenlg": (
        "Please generate high-level steps to create a recipe for the specified food. Don't include "
        "extraneous commentary, or examples, but do refer to the special characteristics and state "
        "of the ingredients used in each step. Don't produce any text other than the list of "
        "steps. Use any of the provided reference recipes to answer the question."
    ),
    "champ": (
        "Please generate high-level steps to solve the given math problem. Don't include code, "
        "extraneous commentary, or examples, but do refer to the concepts and hints used in each "
        "step. Don't produce any text other than the list of steps. Use any of the provided "
        "similar problems and solutions to answer the question."
    ),
}
rag_ex_names: dict[str, str] = {
    "lcstep": "DOCUMENTATION",
    "recipenlg": "RECIPE",
    "champ": "EXAMPLE",
}
rag_inst = {
    "lcstep": (
        "Please generate a list of instructions to accomplish '{query}' using the procedures "
        "above. Create and use these resources in your response: {_input}."
    ),
    "recipenlg": (
        "Please generate a list of instructions to accomplish '{query}' using the recipes above. "
        "Use these ingredients in your response: {_input}."
    ),
    "champ": (
        "Please generate a list of instructions to solve '{query}' using the examples above. Use "
        "this additional information in preparing your response: {_input}."
    ),
}


class RAG(System):
    """This model prompts an LM to generate text few-shot, with the examples provided by searching a
    vector store for texts with similar embeddings to the title."""

    model: Model
    docs: weaviate.collections.Collection
    k: int
    dataset: str

    def __init__(self, model: Model, docs: weaviate.collections.Collection, k: int, dataset: str):
        self.model = model
        self.docs = docs
        self.k = k
        self.dataset = dataset

    def generate(self, query: str, _input: str) -> Response:
        out = self._prepare_result(query, _input)
        completion = self.model.generate(out.prompt)[0]
        out.answer = self.parse_completion(completion)

        return out

    async def agenerate(self, query: str, _input: str) -> Response:
        out = self._prepare_result(query, _input)

        completion = (await self.model.agenerate(out.prompt))[0]
        out.answer = self.parse_completion(completion)
        out.output_tokens = self.model.get_num_tokens(completion)

        return out

    def _prepare_result(self, query: str, _input: str) -> Response:
        """Do everything except get the completion (which can be async)"""
        query_using_input = f"{query} using {_input}"

        docs = self.get_docs(query_using_input)
        context = self.build_context(docs)
        prompt = self.build_prompt(query, _input, context)

        return Response(
            answer=[],  # not set yet
            prompt=prompt,
            model=self.model.name,
            retrieved_docs=docs,
            context=context,
            input_tokens=self.model.get_num_tokens(prompt),
            output_tokens=0,  # not set yet
        )

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

    def build_context(self, docs: list[Doc]) -> str:
        out = ""
        for doc in docs:
            out += f"\n\n{rag_ex_names[self.dataset]} '{doc.title}':\n\n{doc.contents}"

        return out[2:]  # skip first "\n\n"

    def build_prompt(self, query: str, _input: str, context: str) -> str | list[BaseMessage]:
        msg_prompt = context + "\n\n" + rag_inst[self.dataset].format(query=query, _input=_input)

        return self.model.build_prompt(msg_prompt, rag_task[self.dataset])
