import logging

import weaviate
import weaviate.classes as wvc
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

from . import utils
from .interface import Result, System
from .model import Model


_docs_collection = "Docs"


def setup_store(
    logger: logging.Logger, store: weaviate.WeaviateClient, docs: list[dict[str, str | int]]
) -> weaviate.Collection:
    """Create a vector store with the provided docs."""
    if store.collections.exists(_docs_collection):
        logger.info("reusing existing Weaviate collection")
        out = store.collections.get(_docs_collection)
    else:
        logger.info("creating new Weaviate collection")
        out = store.collections.create(
            name=_docs_collection,
            description="Documentation for the LangChain Python library.",
            vectorizer_config=wvc.Configure.Vectorizer.text2vec_transformers(),
            properties=[
                wvc.Property(
                    name="title",
                    data_type=wvc.DataType.TEXT,
                    description="The title of the document",
                ),
                wvc.Property(
                    name="chunk",
                    data_type=wvc.DataType.INT,
                    description="Zero-indexed chunk number in the document",
                    skip_vectorization=True,
                    vectorize_property_name=False,
                ),
                wvc.Property(
                    name="contents",
                    data_type=wvc.DataType.TEXT,
                    description="The contents of (this chunk of) the document",
                ),
            ],
        )

    if len(out) == 0:
        logger.info(f"uploading {len(docs)} chunks to Weaviate collection")
        utils.weaviate_insert(logger, out, docs)
    else:
        logger.info(f"using old data, {len(out)} chunks")

    return out


class RAG(System):
    """This model prompts an LM to generate text few-shot, with the examples provided by searching a
    vector store for texts with similar embeddings to the title."""

    model: Model
    docs: weaviate.Collection
    k: int
    instructions: str = (
        "Please generate high-level steps to accomplish the specified goal using the LangChain "
        "Python library. Don't include code, extraneous commentary, or examples, but do refer to "
        "the specific LangChain APIs (or other APIs) used in each step. Don't produce any text "
        "other than the list of steps. Use any of the provided reference documentation to answer "
        "the question."
    )

    def __init__(
        self,
        model: Model,
        docs: weaviate.Collection,
        k: int,
    ):
        self.model = model
        self.docs = docs
        self.k = k

    def generate(self, query: str) -> Result:
        out = self._prepare_result(query)
        out.answers = self.model.generate(out.prompt)

        return out

    async def agenerate(self, query: str) -> Result:
        out = self._prepare_result(query)
        out.answers = await self.model.agenerate(out.prompt)

        return out

    def _prepare_result(self, query: str) -> Result:
        """Do everything except get the completions (which can be async)"""
        docs = self.get_docs(query)
        context = self.build_context(docs)
        prompt = self.build_prompt(query, context)

        return Result(
            query=query,
            prompt=prompt,
            answers=[],
            model=self.model.model,
            retrieved_docs=[{"path": doc[0], "contents": doc[1]} for doc in docs],
            context=context,
        )

    def get_docs(self, title: str) -> list[tuple[str, str]]:
        """Returns the docs that will be inserted into the prompt."""
        res = self.docs.query.near_text(
            query=title, limit=self.k, return_properties=["title", "contents"]
        )

        out: list[tuple[str, str]] = []
        for obj in res.objects:
            out.append((obj.properties["title"], obj.properties["contents"]))

        return out

    def build_context(self, docs: list[tuple[str, str]]) -> str:
        out = ""
        for t, doc in docs:
            out += f"\n\nDOCUMENTATION FOR '{t}':\n\n{doc}"

        return out

    def build_prompt(self, title: str, context: str) -> list[BaseMessage]:
        msg = (
            f"Please generate a list of instructions to accomplish '{title}' using the "
            "documentation below:"
        )
        msg += context

        return [
            SystemMessage(content=self.instructions),
            HumanMessage(content=msg),
        ]
