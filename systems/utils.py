import logging
from typing import Any
import pickle
import weaviate
from weaviate import classes as wvc
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

def get_vector_representation(logger: logging.Logger, queries: list[str]):
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    return model.encode(queries).tolist()

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

def populate_vector_store(logger: logging.Logger, client: weaviate.WeaviateClient, docs: dict[str, list[str]], collection_name: str):
    if not os.path.exists("../dataset/LCStep/obj_list.pkl"):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
        store_obj_list = []
        for ref_doc in docs.iter(1):
            split_ref_doc = text_splitter.split_text(ref_doc["contents"][0])
            vector_reps = get_vector_representation(logger, split_ref_doc)
            for idx, text_content in enumerate(split_ref_doc):
                store_obj_list.append(wvc.data.DataObject(
                properties={
                    "title": ref_doc["title"][0],
                    "chunk": idx,
                    "contents": text_content,
                },
                vector=vector_reps[idx]
                ))
        
        
        with open("../dataset/LCStep/obj_list.pkl", 'wb') as f:
            pickle.dump(store_obj_list, f)
            f.close()
    else:
        with open("../dataset/LCStep/obj_list.pkl", 'rb') as f:
            store_obj_list = pickle.load(f)

    store = client.collections.get(collection_name)
    weaviate_insert(logger, store, store_obj_list)