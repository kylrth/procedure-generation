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


def chunk_docs(
    docs: list[dict[str, str]], chunk_size: int = 4000, overlap: int = 200
) -> list[dict[str, str | int]]:
    """Takes a list of docs (each should have keys "title" and "contents"), and returns a chunked
    version.

    The output dicts have keys "title", "chunk" (0-indexed), and "contents".
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    chunks = []
    for doc in docs:
        split_doc = text_splitter.split_text(doc["contents"])
        for idx, text_content in enumerate(split_doc):
            chunks.append(
                {
                    "title": doc["title"],
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
                vector=v,
            )
        )

    res = c.data.insert_many(objects)
    if res.has_errors:
        if len(res.errors) > 0:
            logger.error("first Weaviate error: " + next(iter(res.errors.values())).message)

        raise ValueError(f"{len(res.errors)} errors while inserting to Weaviate")


_lcstep_rag_cache = Path("./cache/data/lcstep")
_lcstep_rag_json = _lcstep_rag_cache / "prepared.json"
_lcstep_rag_vectors = _lcstep_rag_cache / "vectors.npy"


def populate(
    logger: logging.Logger,
    store: weaviate.collections.Collection,
    docs: list[dict[str, str]],
):
    """Chunk, vectorize, cache, and upload the docs.

    Each doc dict must have keys "title" and "contents".
    """
    if not _lcstep_rag_cache.exists():
        logger.debug("chunking %d documents", len(docs))
        chunks = chunk_docs(docs, chunk_size=3000)

        logger.debug("vectorizing %d chunks", len(chunks))
        vectors = get_embeds([c["title"] + "\n\n" + c["contents"] for c in chunks], progress=True)

        # cache prepared data and vectors
        _lcstep_rag_cache.mkdir(parents=True, exist_ok=True)
        with _lcstep_rag_json.open("w") as f:
            json.dump(chunks, f)
        np.save(_lcstep_rag_vectors, vectors)
    else:
        # load cached data and vectors
        with _lcstep_rag_json.open("r") as f:
            chunks = json.load(f)
        vectors = np.load(_lcstep_rag_vectors)

    weaviate_insert(logger, store, chunks, vectors)
