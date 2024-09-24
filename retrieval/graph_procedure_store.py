import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from os import PathLike
from typing import Sequence, cast

import numpy as np
import weaviate
import weaviate.classes as wvc
import weaviate.classes.config as wc
from weaviate.collections.classes.batch import BatchObjectReturn, BatchReferenceReturn
from weaviate.types import UUID

from retrieval.embedder import CachingEmbedder, embedder_from_name
from utils import spread_gather


class Node[T]:
    data: T
    incoming: list["Edge"]
    outgoing: list["Edge"]


class Edge[T]:
    output: str
    to: Node[T] | None
    from_: Node[T] | None

    def __init__(self, output: str, to: Node[T], from_: Node[T]):
        self.output = output
        self.to = to
        self.from_ = from_


class Input[T](Edge[T]):
    from_ = None

    def __init__(self, output: str, to: Node[T]):
        self.output = output
        self.to = to


class Output[T](Edge[T]):
    to = None

    def __init__(self, output: str, from_: Node[T]):
        self.output = output
        self.from_ = from_


class Graph[T]:
    inputs: list[Input[T]]
    outputs: list[Output[T]]


@dataclass
class Step:
    api: str
    desc: str
    args: list[str]


class Procedure(Graph[Step], ABC):
    """A graph of steps to accomplish a given task."""

    @abstractmethod
    def __str__(self) -> str:
        """Procedure types must implement a formatting method."""


class Recipe(Procedure):
    def __str__(self) -> str:
        raise NotImplementedError


class LangChainProcedure(Procedure):
    def __str__(self) -> str:
        raise NotImplementedError


class MathSolution(Procedure):
    def __str__(self) -> str:
        raise NotImplementedError


class GraphProcedureStore:
    def __init__(
        self, store: weaviate.WeaviateAsyncClient, embedder: str, cache_path: str | PathLike
    ):
        self.store = store
        self.embedder = CachingEmbedder(embedder_from_name(embedder), cache_path)

    async def setup_collection(self):
        # node collection
        self.nodes = await self.store.collections.create(
            name="StepNode",
            description="A node in a procedure graph",
            properties=[
                wc.Property(
                    name="api", data_type=wc.DataType.TEXT, description="A short title for the step"
                ),
                wc.Property(
                    name="description",
                    data_type=wc.DataType.TEXT,
                    description="A natural-language description of the step, in-context.",
                ),
                wc.Property(
                    name="args",
                    data_type=wc.DataType.TEXT_ARRAY,
                    description="Arguments to the action that affect its behavior.",
                ),
            ],
        )

        # edge collection
        self.edges = await self.store.collections.create(
            name="Edge",
            description="An input or output for a procedure or step",
            properties=[
                wc.Property(
                    name="contents",
                    data_type=wc.DataType.TEXT,
                    description="The name of the object",
                ),
            ],
            # edges track references to nodes
            references=[
                wc.ReferenceProperty(name="toNode", target_collection="StepNode"),
                wc.ReferenceProperty(name="fromNode", target_collection="StepNode"),
            ],
        )

        # nodes track references to edges
        await self.nodes.config.add_reference(
            wc.ReferenceProperty(name="incoming", target_collection="Edge")
        )
        await self.nodes.config.add_reference(
            wc.ReferenceProperty(name="outgoing", target_collection="Edge")
        )

        # graph collection
        self.graphs = await self.store.collections.create(
            name="Procedure",
            description="Procedure graphs",
            properties=[],
            references=[
                wc.ReferenceProperty(name="inputs", target_collection="Edge"),
                wc.ReferenceProperty(name="outputs", target_collection="Edge"),
            ],
        )

    async def populate(self, logger: logging.Logger, procs: list[Procedure]):
        # embed
        logger.debug("embedding %d graph procedures", len(procs))
        formatted = [str(p) for p in procs]
        vectors = await self.embedder.embed(formatted)

        # empty embeddings cache to disk because we likely won't need them during generation
        self.embedder.flush()

        # insert to Weaviate
        async def _task(g_v: tuple[Procedure, np.ndarray]):
            await self.add_graph(logger, g_v[0], g_v[1])

        use_tqdm = logger.getEffectiveLevel() >= logging.DEBUG
        await spread_gather(
            _task, zip(procs, vectors), n=10, length=len(procs) if use_tqdm else None
        )

    def _raise_errors(self, logger: logging.Logger, res: BatchObjectReturn | BatchReferenceReturn):
        if not res.has_errors:
            return

        if len(res.errors) > 0:
            logger.error("first Weaviate error: " + next(iter(res.errors.values())).message)

        raise ValueError(f"{len(res.errors)} errors while inserting to Weaviate")

    async def _insert_edges(
        self,
        logger: logging.Logger,
        edges: Sequence[Edge[Step]],
        seen: dict[int, UUID],
        prev_uuids: Sequence[UUID | None] | None = None,
    ) -> Sequence[UUID]:
        if prev_uuids is None:
            prev_uuids = [None] * len(edges)

        for e in edges:
            if id(e) in seen:
                raise ValueError("malformed graph: encountered the same edge twice")
        if len({id(e) for e in edges}) != len(edges):
            raise ValueError("malformed graph: encountered the same edge twice")

        res = await self.edges.data.insert_many(
            [
                wvc.data.DataObject(
                    properties={"contents": e.output},
                    references={"toNode": uuid} if uuid is not None else None,
                )
                for e, uuid in zip(edges, prev_uuids, strict=True)
            ]
        )
        self._raise_errors(logger, res)
        edge_uuids = [res.uuids[i] for i in range(len(edges))]

        # insert back-references for nodes as well
        refs = []
        for i in range(len(prev_uuids)):
            if prev_uuids[i] is None:
                continue
            refs.append(
                wvc.data.DataReference(
                    from_uuid=cast(UUID, prev_uuids[i]),
                    from_property="incoming",
                    to_uuid=edge_uuids[i],
                )
            )
        if refs:
            res = await self.nodes.data.reference_add_many(refs)
            self._raise_errors(logger, res)

        # mark as seen and store UUIDs
        for e, uuid in zip(edges, edge_uuids, strict=True):
            seen[id(e)] = uuid

        return edge_uuids

    async def _insert_nodes(
        self,
        logger: logging.Logger,
        nodes: Sequence[Node[Step]],
        seen: dict[int, UUID],
        prev_uuids: Sequence[UUID | None] | None = None,
    ) -> Sequence[UUID]:
        if prev_uuids is None:
            prev_uuids = [None] * len(nodes)

        # insert nodes along with references to the outgoing edges if the UUIDs were provided
        new_nodes = []
        only_refs = []
        for n, uuid in zip(nodes, prev_uuids, strict=True):
            if id(n) in seen:
                # we've seen this node before, but we still need to add references to the outgoing
                # edge we discovered it through
                if uuid is not None:
                    only_refs.append(
                        wvc.data.DataReference(
                            from_uuid=seen[id(n)],
                            from_property="outgoing",
                            to_uuid=uuid,
                        )
                    )
            else:
                new_nodes.append(
                    wvc.data.DataObject(
                        properties={
                            "api": n.data.api,
                            "description": n.data.desc,
                            "args": n.data.args,
                        },
                        references={"outgoing": uuid} if uuid is not None else None,
                    )
                )
        # new nodes
        res = await self.nodes.data.insert_many(new_nodes)
        self._raise_errors(logger, res)
        node_uuids = [res.uuids[i] for i in range(len(nodes))]
        # forward references for existing nodes
        res = await self.nodes.data.reference_add_many(only_refs)
        self._raise_errors(logger, res)

        # insert back-references for edges as well
        refs = []
        for i in range(len(prev_uuids)):
            if prev_uuids[i] is None:
                continue
            refs.append(
                wvc.data.DataReference(
                    from_uuid=cast(UUID, prev_uuids[i]),
                    from_property="fromNode",
                    to_uuid=node_uuids[i],
                )
            )
        if refs:
            res = await self.edges.data.reference_add_many(refs)
            self._raise_errors(logger, res)

        return node_uuids

    async def _insert_graph(
        self, input_uuids: Sequence[UUID], output_uuids: Sequence[UUID], v: np.ndarray
    ) -> UUID:
        uuid = await self.graphs.data.insert(
            properties={},
            references={"inputs": input_uuids, "outputs": output_uuids},
            vector=v.tolist(),
        )

        return uuid

    async def add_graph(self, logger: logging.Logger, g: Procedure, v: np.ndarray):
        # we'll traverse the graph starting from the outputs, adding nodes and edges to their
        # collections and setting up references
        seen_edges: dict[int, UUID] = {}
        seen_nodes: dict[int, UUID] = {}

        # start with outputs
        output_uuids = await self._insert_edges(logger, g.outputs, seen_edges)

        # discover the nodes these outputs come from
        next_nodes: list[Node[Step]] = []
        for e in g.outputs:
            if e.from_ is None:
                raise ValueError("malformed graph: output edge did not have from node")
            next_nodes.append(e.from_)

        input_uuids: list[UUID] = []

        prev_uuids = output_uuids
        while next_nodes:
            # add any new nodes, and create references to the previous edges
            prev_uuids = await self._insert_nodes(logger, next_nodes, seen_nodes, prev_uuids)
            # TODO bug here: if some nodes had been seen, we need to skip following their incoming
            # edges
            #
            # TODO next we need to test this well :(

            # discover the next edges
            next_edges: list[Edge[Step]] = []
            for n in next_nodes:
                next_edges.extend(n.incoming)

            # add these edges, and create references to the previous nodes
            prev_uuids = await self._insert_edges(logger, next_edges, seen_edges, prev_uuids)

            # discover the next nodes
            next_nodes = []
            for i, e in enumerate(next_edges):
                if e.from_ is None:
                    # must be input edge
                    input_uuids.append(prev_uuids[i])
                    continue

                next_nodes.append(e.from_)

        for e in g.inputs:
            if id(e) not in seen_edges:
                raise ValueError("malformed graph: input edge not reached by backward traversal")

        # insert graph
        await self._insert_graph(input_uuids, output_uuids, v)
