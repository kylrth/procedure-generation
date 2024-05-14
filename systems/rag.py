import logging

import weaviate
import weaviate.classes.config as wc
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

import retrieval

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


class RAG(System):
    """This model prompts an LM to generate text few-shot, with the examples provided by searching a
    vector store for texts with similar embeddings to the title."""

    model: Model
    docs: weaviate.collections.Collection
    k: int
    dataset: str
    instructions: dict[str, str] = {
        "lcstep": (
            "Please generate high-level steps to accomplish the specified goal using the LangChain "
            "Python library. Don't include code, extraneous commentary, or examples, but do refer to "
            "the specific LangChain APIs (or other APIs) used in each step. Don't produce any text "
            "other than the list of steps. Use any of the provided reference documentation to answer "
            "the question."
        ),
        "recipenlg": (
            "Please generate high-level steps to accomplish the specified goal "
            ". Don't include extraneous commentary, or examples, but do refer to "
            "the special characteristics and state of the ingredients used in each step. Don't produce any text "
            "other than the list of steps. Use any of the provided reference documentation to answer "
            "the question."
        ),
        "champ": (
            "Please generate high-level steps to accomplish the specified goal "
            ". Don't include code, extraneous commentary, or examples, but do refer to "
            "the concepts and hints used in each step. Don't produce any text "
            "other than the list of steps. Use any of the provided reference documentation to answer "
            "the question."
        ),
    }

    def __init__(self, model: Model, docs: weaviate.collections.Collection, k: int, dataset: str):
        self.model = model
        self.docs = docs
        self.k = k
        self.dataset = dataset

    def generate(self, query: str, inp_info: str) -> Response:
        out = self._prepare_result(query, inp_info)
        out.answer = self.model.generate(out.prompt)[0]

        return out

    async def agenerate(self, query: str, inp_info: str) -> Response:
        out = self._prepare_result(query, inp_info)
        out.answer = (await self.model.agenerate(out.prompt))[0]

        return out

    def _prepare_result(self, query: str, inp_info: str) -> Response:
        """Do everything except get the completion (which can be async)"""
        docs = self.get_docs(query)
        context = self.build_context(docs)
        prompt = self.build_prompt(query, inp_info, context)

        return Response(
            query=query,
            prompt=prompt,
            answer="",  # not set yet
            model=self.model.model,
            retrieved_docs=[{"path": doc[0], "contents": doc[1]} for doc in docs],
            context=context,
        )

    def get_docs(self, title: str) -> list[tuple[str, str]]:
        """Returns the docs that will be inserted into the prompt."""
        embedded_query = retrieval.get_embeds([title])[0]

        res = self.docs.query.near_vector(
            near_vector=embedded_query.tolist(),
            limit=self.k,
            return_properties=["title", "contents"],
        )

        out: list[tuple[str, str]] = []
        for obj in res.objects:
            out.append((obj.properties["title"], obj.properties["contents"]))

        return out

    def build_context(self, docs: list[tuple[str, str]]) -> str:
        out = ""
        for title, contents in docs:
            out += f"\n\nDOCUMENTATION FOR '{title}':\n\n{contents}"

        return out

    def build_prompt(self, title: str, inp_info: str, context: str) -> list[BaseMessage]:
        msg = {
            "lcstep": (
                f"Please generate a list of instructions to accomplish '{title}' using the "
                f"documentation below. You have to create and use these resources in your response: {inp_info}"
            ),
            "recipenlg": (
                f"Please generate a list of instructions to accomplish '{title}' using the "
                f"documentation below. You are expected to use these ingredients in your response: {inp_info}"
            ),
            "champ": (
                f"Please generate a list of instructions to accomplish '{title}' using the "
                f"documentation below. You are expected to use this additional information in preparing your response: {inp_info}"
            ),
        }
        msg_prompt = msg[self.dataset]
        msg_prompt += context

        return [
            SystemMessage(content=self.instructions[self.dataset]),
            HumanMessage(content=msg_prompt),
        ]
