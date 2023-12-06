import logging

import weaviate
import weaviate.classes as wvc
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

from .interface import System
from .model import Model, log


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
        res = out.data.insert_many(docs)
        if res.has_errors:
            if len(res.errors) > 0:
                logger.error("first Weaviate error: " + next(iter(res.errors.values())).message)

            raise ValueError(f"there were {len(res.errors)} upload errors")
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
        "other than the list of steps. Use the provided reference documentation to answer the "
        "question."
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

    def generate(self, title: str, logger: logging.Logger) -> list[str]:
        docs = self.get_docs(title, logger)
        prompt = self.build_prompt(title, docs)
        completion = self.model.generate(prompt)

        log(logger, "RAG", prompt, completion)

        return completion

    async def agenerate(self, title: str, logger: logging.Logger) -> list[str]:
        docs = self.get_docs(title, logger)
        prompt = self.build_prompt(title, docs)
        completion = await self.model.agenerate(prompt)

        log(logger, "RAG", prompt, completion)

        return completion

    def get_docs(self, title: str, logger: logging.Logger) -> list[tuple[str, str]]:
        """Returns the docs that will be inserted into the prompt."""
        res = self.docs.query.near_text(
            query=title, limit=self.k, return_properties=["title", "contents"]
        )

        out: list[tuple[str, str]] = []
        for obj in res.objects:
            out.append((obj.properties["title"], obj.properties["contents"]))

        logger.debug(f"retrieved {len(out)} docs for query '{title}'")

        return out

    def build_prompt(self, title: str, docs: list[tuple[str, str]]) -> list[BaseMessage]:
        msg = (
            f"Please generate a list of instructions to accomplish '{title}' using the "
            "documentation below:"
        )
        for t, doc in docs:
            msg += f"\n\nDOCUMENTATION FOR '{t}':\n\n{doc}"

        return [
            SystemMessage(content=self.instructions),
            HumanMessage(content=msg),
        ]
