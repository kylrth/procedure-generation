import logging
from typing import Any

import weaviate
from weaviate import classes as wvc


def weaviate_insert(
    logger: logging.Logger,
    c: weaviate.Collection,
    data: list[dict[str, Any] | wvc.DataObject[dict[str, Any]]],
):
    res = c.data.insert_many(data)
    if res.has_errors:
        if len(res.errors) > 0:
            logger.error("first Weaviate error: " + next(iter(res.errors.values())).message)

        raise ValueError(f"{len(res.errors)} errors while inserting to Weaviate")
