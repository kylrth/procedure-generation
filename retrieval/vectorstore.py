import itertools
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import weaviate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from weaviate import classes as wvc

from dataset import Doc


def chunk_docs(
    docs: list[Doc], chunk_size: int = 4000, overlap: int = 200
) -> list[dict[str, str | int]]:
    """Takes a list of docs (each should have keys "title" and "contents"), and returns a chunked
    version.

    The output dicts have keys "title", "chunk" (0-indexed), and "contents".
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    chunks = []
    for doc in docs:
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


_embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def get_embeds(queries: list[str], progress: bool = False) -> list[np.ndarray]:
    return _embedder.encode(queries, show_progress_bar=progress, convert_to_numpy=True)


def weaviate_insert(
    logger: logging.Logger,
    c: weaviate.collections.Collection,
    properties: list[dict[str, Any]],
    vectors: list[np.ndarray] | None,
):
    objects = []
    for d, v in zip(
        properties,
        vectors if vectors is not None else itertools.repeat(None, len(properties)),
        strict=True,
    ):
        objects.append(
            wvc.data.DataObject(
                properties=d,
                vector=v.tolist() if v is not None else None,
            )
        )

    res = c.data.insert_many(objects)
    if res.has_errors:
        if len(res.errors) > 0:
            logger.error("first Weaviate error: " + next(iter(res.errors.values())).message)

        raise ValueError(f"{len(res.errors)} errors while inserting to Weaviate")


_rag_cache = "./cache/vectors/{name}"


def populate(
    logger: logging.Logger, name: str, store: weaviate.collections.Collection, docs: list[Doc]
):
    """Chunk, vectorize, cache, and upload the docs.

    Each doc dict must have keys "title" and "contents".
    """
    rag_cache = Path(_rag_cache.format(name=name))

    if not rag_cache.exists():
        logger.debug("chunking %d documents", len(docs))
        chunks = chunk_docs(docs, chunk_size=3000)

        logger.debug("vectorizing %d chunks", len(chunks))
        vectors = get_embeds([c["title"] + "\n\n" + c["contents"] for c in chunks], progress=True)

        # cache prepared data and vectors
        rag_cache.mkdir(parents=True, exist_ok=True)
        with (rag_cache / "prepared.json").open("w") as f:
            json.dump(chunks, f)
        np.save(rag_cache / "vectors.npy", vectors)
    else:
        logger.debug("loading cached chunks and vectors")
        # load cached data and vectors
        with (rag_cache / "prepared.json").open("r") as f:
            chunks = json.load(f)
        vectors = np.load(rag_cache / "vectors.npy")

    weaviate_insert(logger, store, chunks, vectors)
