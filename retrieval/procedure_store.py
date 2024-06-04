import asyncio
import logging
from dataclasses import dataclass
from os import PathLike
from typing import ClassVar

import weaviate
import weaviate.classes.config as wc

from dataset import Procedure

from .store import Store


@dataclass
class ProcedureFormatter:
    output_prefix: str
    input_prefix: str

    def format(self, p: Procedure) -> str:
        return (
            self.output_prefix + p.output + self.input_prefix + p.input_ + "\n\n" + p.format_steps()
        )


def procedure_formatter_for(dataset: str) -> ProcedureFormatter:
    if dataset == "lcstep":
        return ProcedureFormatter(
            output_prefix="instructions to ",
            input_prefix=" given ",
        )
    if dataset == "recipenlg":
        return ProcedureFormatter(
            output_prefix="a recipe for ",
            input_prefix="\n\ningredients: ",
        )
    if dataset == "champ":
        return ProcedureFormatter(
            output_prefix="Solve the following problem:\n\n",
            input_prefix="\n\nYou may use these hints: ",
        )

    raise NotImplementedError(dataset)


class ProcedureStore(Store):
    _coll_properties: ClassVar[list[wc.Property]] = [
        wc.Property(
            name="input",
            data_type=wc.DataType.TEXT,
            description="The inputs used by this procedure",
        ),
        wc.Property(
            name="output",
            data_type=wc.DataType.TEXT,
            description="The output created by this procedure",
        ),
        wc.Property(
            name="steps",
            data_type=wc.DataType.TEXT_ARRAY,
            description="The list of steps to produce the output from the input",
        ),
    ]

    pfmt: ProcedureFormatter

    def __init__(
        self,
        store: weaviate.WeaviateClient,
        name: str,
        desc: str,
        embedder: str,
        cache_path: str | PathLike,
        pfmt: ProcedureFormatter,
    ):
        super().__init__(store, name, desc, embedder, cache_path)
        self.pfmt = pfmt

    @property
    def coll_properties(self) -> list[wc.Property]:
        return self._coll_properties

    def populate(self, logger: logging.Logger, procs: list[Procedure]):
        """Chunk, vectorize, cache, and upload the procedures."""
        logger.debug("embedding %d procedures", len(procs))
        formatted = [self.pfmt.format(p) for p in procs]
        vectors = asyncio.run(self.embedder.embed(formatted))

        # empty embeddings cache to disk because we likely won't need them during generation
        self.embedder.flush()

        self.weaviate_insert(logger, [p.to_dict() for p in procs], vectors)

    async def search(self, query: str, k: int = 5) -> list[Procedure]:
        """Returns the procedures that will be inserted into the prompt."""
        embedded_query = (await self.embedder.embed([query]))[0]

        res = self.collection.query.near_vector(
            near_vector=embedded_query.tolist(),
            limit=k,
            return_properties=["input", "output", "steps"],
        )

        out = []
        for obj in res.objects:
            out.append(
                Procedure(
                    obj.properties["input"], obj.properties["output"], obj.properties["steps"]
                )
            )

        return out
