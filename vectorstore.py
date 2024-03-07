import logging
import pickle
from pathlib import Path

import weaviate
from datasets import Dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from weaviate import classes as wvc

_embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def get_vector_representation(queries: list[str], progress: bool = False):
    return _embedder.encode(queries, show_progress_bar=progress).tolist()


def weaviate_insert(
    logger: logging.Logger,
    c: weaviate.Collection,
    data: wvc.data.DataObject,
):
    res = c.data.insert_many(data)
    if res.has_errors:
        if len(res.errors) > 0:
            logger.error("first Weaviate error: " + next(iter(res.errors.values())).message)

        raise ValueError(f"{len(res.errors)} errors while inserting to Weaviate")


_lcstep_vector_cache = Path("./dataset/LCStep/obj_list.pkl")


def populate(
    logger: logging.Logger,
    client: weaviate.WeaviateClient,
    docs: Dataset,
    collection_name: str,
):
    if not _lcstep_vector_cache.exists():
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)

        chunks = []
        for ref_doc in docs.iter(1):
            split_ref_doc = text_splitter.split_text(ref_doc["contents"][0])
            for idx, text_content in enumerate(split_ref_doc):
                chunks.append(
                    {
                        "title": ref_doc["title"][0],
                        "chunk": idx,
                        "contents": text_content,
                    }
                )

        logger.debug("vectorizing %d chunks", len(chunks))
        vectors = get_vector_representation(chunks, progress=True)
        vectors = iter(vectors)

        objects = []
        for d in chunks:
            objects.append(
                wvc.data.DataObject(
                    properties=d,
                    vector=next(vectors),
                )
            )

        with _lcstep_vector_cache.open("wb") as f:
            pickle.dump(objects, f)
    else:
        with _lcstep_vector_cache.open("rb") as f:
            objects = pickle.load(f)

    store = client.collections.get(collection_name)
    weaviate_insert(logger, store, objects)
