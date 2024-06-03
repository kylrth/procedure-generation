import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from langchain.schema import BaseMessage
import weaviate
from dataset import Doc, Procedure
import logging
import itertools
from weaviate import classes as wvc
from typing import Any
from embedder import embedder_from_name, OpenAIEmbedder, HFEmbedder

class Store(ABC):
    """This is the interface that all systems in this package will implement. It can be imported for
    type annotations."""
    colln_name: str
    colln_desc: str
    store: weaviate.WeaviateClient
    embedder:Any
    
    def __init__(self, store: weaviate.WeaviateClient, name: str, desc: str, embedder: str):
        self.store = store
        self.name = name
        self.desc = desc
        self.embedder = embedder_from_name(embedder)

    @abstractmethod
    def setup_store(self, logger: logging.Logger) -> weaviate.collections.Collection:
        pass

    @abstractmethod
    def populate(self, logger: logging.Logger, docs: list[Procedure]| list[Doc])->weaviate.collections.Collection:
        pass
    
    @abstractmethod
    def get_docs(self, query: str) -> list[Doc] | list[Procedure]:
        
    
    @abstractmethod
    def format_str(self, text: Procedure| dict[str, str | int])->str:
        pass
    
    def get_embeds(self, data: list[dict[str, str | int]] | list[Procedure])->list[np.ndarray]:
        queries = [self.format_str(data_elem) for data_elem in data]
        embeddings = self.embedder.embed(queries)
        return embeddings
    
    def weaviate_insert(
        self,
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