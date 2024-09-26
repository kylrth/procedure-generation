import itertools
import logging
from abc import ABC, abstractmethod
from os import PathLike
from typing import Any, Type, cast

import numpy as np
import weaviate
from weaviate import classes as wvc

from .embedder import CachingEmbedder, embedder_from_name


class Store(ABC):
    """This base class provides utility methods for custom data storage objects backed by Weaviate.

    Store subclasses should implement a `populate` method that adds data to the collection.
    """

    store: weaviate.WeaviateAsyncClient
    collection: weaviate.collections.CollectionAsync
    embedder: CachingEmbedder

    @classmethod
    async def new[
        T
    ](
        cls: Type[T],
        store: weaviate.WeaviateAsyncClient,
        name: str,
        desc: str,
        embedder: str,
        cache_path: str | PathLike,
    ) -> T:
        self = cast(Store, cls())

        self.store = store
        self.collection = await self.store.collections.create(
            name=name,
            description=desc,
            properties=self.coll_properties,
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
        )
        self.embedder = CachingEmbedder(embedder_from_name(embedder), cache_path)

        return cast(T, self)

    @property
    @abstractmethod
    def coll_properties(self) -> list[wvc.config.Property]:
        """Subclasses must define the Weaviate properties that will be used to create the Weaviate
        collection."""

    @abstractmethod
    async def search(self, query: str, k: int = 5) -> list[Any]:
        pass

    async def weaviate_insert(
        self,
        logger: logging.Logger,
        properties: list[dict[str, Any]],
        vectors: list[np.ndarray] | None,
    ):
        """Subclasses may use this convenience function to add data to the Weaviate collection."""
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

        res = await self.collection.data.insert_many(objects)
        if res.has_errors:
            if len(res.errors) > 0:
                logger.error("first Weaviate error: " + next(iter(res.errors.values())).message)

            raise ValueError(f"{len(res.errors)} errors while inserting to Weaviate")
