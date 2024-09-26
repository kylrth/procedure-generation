import logging
from abc import ABC, abstractmethod
from typing import Sequence, Type, cast

import numpy as np
import weaviate
import weaviate.classes as wvc
import weaviate.classes.config as wc
from weaviate.classes.query import Filter
from weaviate.collections.classes.batch import BatchObjectReturn, BatchReferenceReturn
from weaviate.types import UUID

from retrieval.embedder import CachingEmbedder, Embedder
from utils import spread_gather


class Node[T]:
    data: T
    incoming: list["Edge[T]"]
    outgoing: list["Edge[T]"]

    def __init__(self, data: T):
        self.data = data
        self.incoming = []
        self.outgoing = []

    def new_edge_to(self, other: "Node[T]", edge: str) -> "Edge[T]":
        out = Edge(edge, other, self)

        self.outgoing.append(out)
        other.incoming.append(out)

        return out

    def add_inputs(self, *inputs: str) -> list["Input[T]"]:
        out = []
        for i in inputs:
            out.append(Input(i, self))

        self.incoming.extend(out)

        return out

    def add_outputs(self, *outputs: str) -> list["Output[T]"]:
        out = []
        for o in outputs:
            out.append(Output(o, self))

        self.outgoing.extend(out)

        return out


class Edge[T]:
    content: str
    to: Node[T] | None
    from_: Node[T] | None

    def __init__(self, content: str, to: Node[T], from_: Node[T]):
        self.content = content
        self.to = to
        self.from_ = from_


class Input[T](Edge[T]):
    from_ = None

    def __init__(self, content: str, to: Node[T]):
        self.content = content
        self.to = to


class Output[T](Edge[T]):
    to = None

    def __init__(self, content: str, from_: Node[T]):
        self.content = content
        self.from_ = from_


class Graph[T]:
    inputs: list[Input[T]]
    outputs: list[Output[T]]

    def __init__(self, *nodes: Node[T]):
        self.inputs = []
        self.outputs = []

        for node in nodes:
            for incoming in node.incoming:
                if isinstance(incoming, Input):
                    self.inputs.append(incoming)
            for outgoing in node.outgoing:
                if isinstance(outgoing, Output):
                    self.outputs.append(outgoing)

    def __eq__(self, ot: object, /) -> bool:
        if not isinstance(ot, Graph):
            return False

        # check output number
        if len(self.outputs) != len(ot.outputs):
            return False
        # check input number
        if len(self.inputs) != len(ot.inputs):
            return False

        self_outputs = sorted(self.outputs, key=lambda x: x.content)
        self_inputs = sorted(self.inputs, key=lambda x: x.content)
        ot_outputs = sorted(ot.outputs, key=lambda x: x.content)
        ot_inputs = sorted(ot.inputs, key=lambda x: x.content)

        # check output contents
        if any(self_outputs[i].content != ot_outputs[i].content for i in range(len(self_outputs))):
            return False
        # check input contents
        if any(self_inputs[i].content != ot_inputs[i].content for i in range(len(self_inputs))):
            return False

        # Note that any graph with no outputs is vacuously the same as any other, because no nodes
        # are reachable via back-traversal.

        visited: set[Node] = set()  # track nodes visited in self
        return all(
            self.__eq_dfs(o1.from_, o2.from_, visited) for o1, o2 in zip(self.outputs, ot.outputs)
        )

    @staticmethod
    def __eq_dfs(o1: Node | None, o2: Node | None, visited: set[Node]) -> bool:  # noqa: PLR0911
        # check if the nodes exist (these might have come from Input edges)
        if o1 is None:
            return o2 is None
        if o2 is None:
            return False

        # first check if the contents are the same
        if o1.data != o2.data:
            return False

        visited.add(o1)

        # We don't check outgoing edges because they're either a) already checked as an incoming
        # edge of some node, or b) an extraneous edge leading to a part of the graph that doesn't
        # produce any output. The latter case is an inconsistent state for a Graph to be in and will
        # be ignored.

        if len(o1.incoming) != len(o2.incoming):
            return False

        for i1, i2 in zip(
            sorted(o1.incoming, key=lambda x: x.content),
            sorted(o2.incoming, key=lambda x: x.content),
        ):
            if i1.content != i2.content:
                return False

            if i1.from_ in visited:
                continue

            # visit nodes
            if not Graph.__eq_dfs(i1.from_, i2.from_, visited):
                return False

        return True


class Step:
    api: str
    desc: str
    args: list[str]

    def __init__(self, api: str, desc: str, args: list[str] | None = None):
        self.api = api
        self.desc = desc
        self.args = args if args is not None else []

    def __eq__(self, other: object, /) -> bool:
        if not isinstance(other, Step):
            return False

        if self.api != other.api:
            return False
        if self.desc != other.desc:
            return False
        return self.args == other.args


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


class GraphProcedureStore[T: Procedure]:
    def __init__(
        self,
        store: weaviate.WeaviateAsyncClient,
        embedder: Embedder,
        cls: Type[T],
    ):
        self.store = store
        self.embedder = embedder
        self.g_cls = cls

    async def setup_collection(self, *, prefix: str = ""):
        # node collection
        self.nodes = await self.store.collections.create(
            name=prefix + "StepNode",
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
            name=prefix + "Edge",
            description="An input or output for a procedure or step",
            properties=[
                wc.Property(
                    name="content",
                    data_type=wc.DataType.TEXT,
                    description="The name of the object",
                ),
            ],
            # edges track references to nodes
            references=[
                wc.ReferenceProperty(name="toNode", target_collection=prefix + "StepNode"),
                wc.ReferenceProperty(name="fromNode", target_collection=prefix + "StepNode"),
            ],
        )

        # nodes track references to edges
        await self.nodes.config.add_reference(
            wc.ReferenceProperty(name="incoming", target_collection=prefix + "Edge")
        )
        await self.nodes.config.add_reference(
            wc.ReferenceProperty(name="outgoing", target_collection=prefix + "Edge")
        )

        # graph collection
        self.graphs = await self.store.collections.create(
            name=prefix + "Procedure",
            description="Procedure graphs",
            properties=[],
            references=[
                wc.ReferenceProperty(name="inputs", target_collection=prefix + "Edge"),
                wc.ReferenceProperty(name="outputs", target_collection=prefix + "Edge"),
            ],
        )

    async def populate(self, logger: logging.Logger, procs: list[T]):
        # embed
        logger.debug("embedding %d graph procedures", len(procs))
        formatted = [str(p) for p in procs]
        vectors = await self.embedder.embed(formatted)

        if isinstance(self.embedder, CachingEmbedder):
            # empty embeddings cache to disk because we likely won't need them during generation
            self.embedder.flush()

        # insert to Weaviate
        async def _task(g_v: tuple[T, np.ndarray]):
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
                    properties={"content": e.content},
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
    ) -> tuple[Sequence[UUID], Sequence[bool]]:
        if prev_uuids is None:
            prev_uuids = [None] * len(nodes)

        # insert nodes along with references to the outgoing edges if the UUIDs were provided
        new_nodes = []
        only_refs = []
        skipped: list[bool] = []
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
                    skipped.append(True)
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
                skipped.append(False)
        # new nodes
        if new_nodes:
            res = await self.nodes.data.insert_many(new_nodes)
            self._raise_errors(logger, res)
            node_uuids = [res.uuids[i] for i in range(len(nodes))]
        else:
            node_uuids = []
        # forward references for existing nodes
        if only_refs:
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

        return node_uuids, skipped

    async def _insert_graph(
        self, input_uuids: Sequence[UUID], output_uuids: Sequence[UUID], v: np.ndarray
    ) -> UUID:
        uuid = await self.graphs.data.insert(
            properties={},
            references={"inputs": input_uuids, "outputs": output_uuids},
            vector=v.tolist(),
        )

        return uuid

    async def add_graph(self, logger: logging.Logger, g: T, v: np.ndarray) -> UUID:  # noqa: C901
        """Add a single graph to the store."""
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
            prev_uuids, skipped = await self._insert_nodes(
                logger, next_nodes, seen_nodes, prev_uuids
            )

            # discover the next edges
            next_edges: list[Edge[Step]] = []
            prev_uuids_filtered = []
            prev_uuids_iter = iter(prev_uuids)
            for n, skip in zip(next_nodes, skipped):
                if skip:  # already saw this node, so we don't need to follow its incoming edges
                    continue
                next_edges.extend(n.incoming)
                prev_uuids_filtered.extend([next(prev_uuids_iter)] * len(n.incoming))

            if not next_edges:
                break

            # add these edges, and create references to the previous nodes
            prev_uuids = await self._insert_edges(
                logger, next_edges, seen_edges, prev_uuids_filtered
            )

            # discover the next nodes
            next_nodes = []
            prev_uuids_filtered = []
            for i, (e, uuid) in enumerate(zip(next_edges, prev_uuids, strict=True)):
                if e.from_ is None:
                    # must be input edge
                    input_uuids.append(prev_uuids[i])
                    continue

                next_nodes.append(e.from_)
                prev_uuids_filtered.append(uuid)
            prev_uuids = prev_uuids_filtered

        for e in g.inputs:
            if id(e) not in seen_edges:
                logger.error("did not see input edge '%s'", e.content)
                raise ValueError("malformed graph: input edge not reached by backward traversal")

        # insert graph
        return await self._insert_graph(input_uuids, output_uuids, v)

    async def get_graph(self, id_: UUID) -> tuple[T, np.ndarray]:
        """Get an existing graph and its embedding by ID."""
        out = self.g_cls()

        # get output edges and nodes they lead from
        res = await self.graphs.query.fetch_object_by_id(
            id_,
            include_vector=True,
            return_properties=[],
            return_references=wvc.query.QueryReference(
                link_on="outputs",
                return_properties=["content"],
                return_references=wvc.query.QueryReference(
                    link_on="fromNode", return_properties=["api", "description", "args"]
                ),
            ),
        )
        v = np.array(res.vector["default"])
        # collect visible nodes and add Output objects to out
        new_nodes: list[UUID] = []
        seen: dict[UUID, Node[Step]] = {}
        for output in res.references["outputs"].objects:
            node_data = output.references["fromNode"].objects[0]
            node = Node(
                Step(
                    cast(str, node_data.properties["api"]),
                    cast(str, node_data.properties["description"]),
                    cast(list[str], node_data.properties["args"]),
                )
            )
            content = cast(str, output.properties["content"])
            out.outputs.extend(node.add_outputs(content))

            new_nodes.append(node_data.uuid)
            seen[node_data.uuid] = node

        while new_nodes:
            # get incoming edges and nodes they lead from
            res = await self.edges.query.fetch_objects(
                filters=Filter.by_ref(link_on="toNode").by_id().contains_any(new_nodes),
                return_properties=["content"],
                return_references=[
                    wvc.query.QueryReference(link_on="toNode"),  # so we know which nodes to link to
                    wvc.query.QueryReference(  # backtrack to new (unseen) nodes
                        link_on="fromNode", return_properties=["api", "description", "args"]
                    ),
                ],
            )

            # collect visible nodes and add Edge objects to out, watching for Input edges
            new_nodes = []
            for edge in res.objects:
                edge_content = cast(str, edge.properties["content"])
                to_node = seen[edge.references["toNode"].objects[0].uuid]

                if "fromNode" not in edge.references:  # Input
                    out.inputs.extend(to_node.add_inputs(edge_content))
                else:  # internal edge
                    node_data = edge.references["fromNode"].objects[0]
                    node = Node(
                        Step(
                            cast(str, node_data.properties["api"]),
                            cast(str, node_data.properties["description"]),
                            cast(list[str], node_data.properties["args"]),
                        )
                    )
                    node.new_edge_to(to_node, edge_content)

                    if node_data.uuid not in seen:
                        new_nodes.append(node_data.uuid)
                    seen[node_data.uuid] = node

        return out, v
