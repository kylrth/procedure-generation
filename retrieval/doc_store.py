import asyncio
import json
import logging
from typing import ClassVar

import weaviate.classes.config as wc
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dataset import Doc

from .store import Store


class DocStore(Store):
    _coll_properties: ClassVar[list[wc.Property]] = [
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
    ]

    @property
    def coll_properties(self) -> list[wc.Property]:
        return self._coll_properties

    def populate(self, logger: logging.Logger, docs: list[Doc]):
        """Chunk, vectorize, cache, and upload the docs."""
        chunk_cache = self.embedder.path / "chunks.json"

        if chunk_cache.exists():
            logger.debug("loading cached chunks")
            with chunk_cache.open("r") as f:
                chunks = json.load(f)
        else:
            logger.debug("chunking %d documents", len(docs))
            chunks = self.chunk_docs(docs, chunk_size=3000)

        logger.debug("embedding %d chunks", len(chunks))
        formatted = [chunk["title"] + "\n\n" + chunk["contents"] for chunk in chunks]
        vectors = asyncio.run(self.embedder.embed(formatted))

        # empty embeddings cache to disk because we likely won't need them during generation
        self.embedder.flush()

        self.weaviate_insert(logger, chunks, vectors)

    def chunk_docs(
        self, data: list[Doc], chunk_size: int = 4000, overlap: int = 200
    ) -> list[dict[str, str | int]]:
        """Takes a list of procedure (each should have keys "input", "output" and "steps"), and
        returns a chunked version.

        The output dicts have keys "title", "chunk" (0-indexed), and "contents".
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

        chunks = []
        for doc in data:
            split_doc = text_splitter.split_text(doc.contents)
            for idx, text_content in enumerate(split_doc):
                chunks.append(
                    {
                        "title": doc.title,
                        "chunk": idx,
                        "contents": text_content,
                    }
                )

        return chunks

    async def search(self, query: str, k: int = 5) -> list[Doc]:
        """Returns the k docs with vector representation closes to that of the query."""
        embedded_query = (await self.embedder.embed([query], is_query=True))[0]

        res = self.collection.query.near_vector(
            near_vector=embedded_query.tolist(),
            limit=k,
            return_properties=["title", "contents"],
        )

        out = []
        for obj in res.objects:
            out.append(Doc(obj.properties["title"], obj.properties["contents"]))

        return out
