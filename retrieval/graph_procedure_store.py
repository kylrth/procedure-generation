import asyncio
import logging
from typing import Sequence, cast

import numpy as np
import weaviate
import weaviate.classes as wvc
import weaviate.classes.config as wc
from weaviate.classes.query import Filter
from weaviate.collections.classes.batch import BatchObjectReturn, BatchReferenceReturn
from weaviate.types import UUID

from dataset import GraphProcedure, Step
from graph import Edge as GEdge
from graph import Graph as GGraph
from graph import Node as GNode
from retrieval.embedder import CachingEmbedder, Embedder
from utils import spread_gather


# For procedures, all our nodes are Steps and our edges are strings.
Edge = GEdge[Step, str]
Graph = GGraph[Step, str]
Node = GNode[Step, str]


class GraphProcedureStore:
    def __init__(
        self,
        store: weaviate.WeaviateAsyncClient,
        embedder: Embedder,
    ):
        self.store = store
        self.embedder = embedder

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

    async def populate(self, logger: logging.Logger, procs: Sequence[GraphProcedure]):
        # embed
        logger.debug("embedding %d graph procedures", len(procs))
        formatted = [str(p) for p in procs]
        vectors = await self.embedder.embed(formatted)

        if isinstance(self.embedder, CachingEmbedder):
            # empty embeddings cache to disk because we likely won't need them during generation
            self.embedder.flush()

        # insert to Weaviate
        async def _task(g_v: tuple[GraphProcedure, np.ndarray]):
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
        edges: Sequence[Edge],
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
        nodes: Sequence[Node],
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

    async def add_graph(  # noqa: C901
        self, logger: logging.Logger, g: GraphProcedure, v: np.ndarray
    ) -> UUID:
        """Add a single graph to the store."""
        # we'll traverse the graph starting from the outputs, adding nodes and edges to their
        # collections and setting up references
        seen_edges: dict[int, UUID] = {}
        seen_nodes: dict[int, UUID] = {}

        # start with outputs
        output_uuids = await self._insert_edges(logger, g.outputs, seen_edges)

        # discover the nodes these outputs come from
        next_nodes: list[Node] = []
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
            next_edges: list[Edge] = []
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

    async def get_graph(self, id_: UUID) -> tuple[GraphProcedure, np.ndarray]:
        """Get an existing graph and its embedding by ID."""
        out = GraphProcedure()

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
        seen: dict[UUID, Node] = {}
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

    async def search(self, query: str, *, k: int = 10) -> list[GraphProcedure]:
        """Find the top k procedures matching the query."""
        v = (await self.embedder.embed([query], is_query=True))[0]
        return await self.search_v(v, k=k)

    async def search_v(self, v: np.ndarray, *, k: int = 10) -> list[GraphProcedure]:
        # find the top matching graphs and just get their UUIDs
        res = await self.graphs.query.near_vector(
            near_vector=v.tolist(), limit=k, return_properties=[]
        )

        # build the graph objects
        gs_vs = await asyncio.gather(*(self.get_graph(obj.uuid) for obj in res.objects))

        return [g for g, _ in gs_vs]

    async def hierarchical_retrieval(
        self, query: str, *, k: int = 10, k2: int = 5
    ) -> list[GraphProcedure]:
        """Of the top k procedures matching the query, extract the top k2 partial procedures.

        Partial procedures are extracted by traversing backward from the outputs, progressively
        cutting out the nodes that are reached and then recomputing the embedding. The closest
        version to the embedding of the original query wins.
        """
        embedded_query = (await self.embedder.embed([query], is_query=True))[0]

        # find the top matching graphs and just get their UUIDs
        res = await self.graphs.query.near_vector(
            near_vector=embedded_query.tolist(), limit=k, return_properties=[]
        )

        # build the graph objects
        gs_vs = await asyncio.gather(*(self.get_graph(obj.uuid) for obj in res.objects))
        graphs, vectors = zip(*gs_vs)

        # collect subgraphs
        subgraphs: list[list[GraphProcedure]] = []
        for graph in graphs:
            these_subgraphs = []

            # collect subgraphs by successively cutting input layers until we reach the empty graph
            sg = graph.copy().cut_input_layer()
            while sg:
                these_subgraphs.append(sg)
                sg = sg.copy().cut_input_layer()

            # now collect subgraphs by cutting output layers
            sg = graph.copy().cut_output_layer()
            while sg:
                these_subgraphs.append(sg)
                sg = sg.copy().cut_output_layer()

        # embed subgraphs (flatten to allow faster batch processing)
        sg_vecs = await self.embedder.embed([str(sg) for sgs in subgraphs for sg in sgs])

        # compute similarity between (sub)graphs and query
        g_sims = np.dot(vectors, embedded_query)  # use dot product as embeddings are normalized
        sg_sims = np.dot(sg_vecs, embedded_query)

        # unflatten subgraph similarity scores
        sg_sims_by_graph: list[np.ndarray] = []
        start = 0
        for sgs in subgraphs:
            sg_sims_by_graph.append(sg_sims[start : start + len(sgs)])
            start += len(sgs)

        # find best (sub)graph of each graph
        bests: list[GraphProcedure] = []
        best_sims: list[float] = []
        start = 0
        for g, sgs, sim, sg_sims in zip(graphs, subgraphs, g_sims, sg_sims_by_graph):
            best_g = g
            best_sim = sim

            best_sg_ind = np.argmax(sg_sims)
            if sg_sims[best_sg_ind] > best_sim:
                best_g = sgs[best_sg_ind]
                best_sim = sg_sims[best_sg_ind]

            bests.append(best_g)
            best_sims.append(best_sim)

        indices = np.argpartition(best_sims, -k2)[-k2:]  # top k2 best subgraphs
        indices = indices[np.argsort(best_sims[indices])]  # indices don't come sorted; sort them

        return [bests[i] for i in indices]
