from abc import abstractmethod
from typing import Protocol, cast


class _Comparable(Protocol):
    @abstractmethod
    def __lt__[U](self: U, value: U, /) -> bool: ...


class Node[T, U: _Comparable]:
    """A node in a Graph, containing data of type T.

    A Node may have no incoming Edges, but it must have at least one outgoing Edge.
    """

    data: T
    incoming: list["Edge[T, U]"]
    outgoing: list["Edge[T, U]"]

    def __init__(self, data: T):
        self.data = data
        self.incoming = []
        self.outgoing = []

    def new_edge_to(self, other: "Node[T, U]", edge: U) -> "Edge[T, U]":
        out = Edge(edge, other, self)

        self.outgoing.append(out)
        other.incoming.append(out)

        return out

    def add_inputs(self, *inputs: U) -> list["Input[T, U]"]:
        out = []
        for i in inputs:
            out.append(Input(i, self))

        self.incoming.extend(out)

        return out

    def add_outputs(self, *outputs: U) -> list["Output[T, U]"]:
        out = []
        for o in outputs:
            out.append(Output(o, self))

        self.outgoing.extend(out)

        return out


class Edge[T, U: _Comparable]:
    """A directed link between two Nodes.

    Edge content must be a comparable type so that Graph equality can ignore Edge listing order.
    """

    content: U
    to: Node[T, U] | None
    from_: Node[T, U] | None

    def __init__(self, content: U, to: Node[T, U], from_: Node[T, U]):
        self.content = content
        self.to = to
        self.from_ = from_


class Input[T, U: _Comparable](Edge[T, U]):
    """A directed input to a Node.

    All Inputs in a Graph should be listed in that graph's inputs field.
    """

    from_ = None

    def __init__(self, content: U, to: Node[T, U]):
        self.content = content
        self.to = to


class Output[T, U: _Comparable](Edge[T, U]):
    """A directed output from a Node.

    All Outputs in a Graph should be listed in that graph's outputs field.
    """

    to = None

    def __init__(self, content: U, from_: Node[T, U]):
        self.content = content
        self.from_ = from_


class Graph[T, U: _Comparable]:
    """A set of nodes of type T and edges of type U, with special Input and Output edges each
    connected to only one Node.

    The Graph is traversible through the inputs and outputs fields. Traversing backward from the
    outputs is guaranteed to discover all nodes in the graph, because Nodes are allowed to have no
    incoming Edges but must have at least one outgoing Edge.

    Two Graphs are equal if their inputs and outputs have the same contents, ignoring list order,
    and if traversing back from the outputs arrives at the same Edge and Node structures with the
    same Edge contents and Node data, again ignoring Edge list order. A Graph with no outputs is the
    empty Graph. The empty Graph is the only Graph whose boolean value is False.
    """

    inputs: list[Input[T, U]]
    outputs: list[Output[T, U]]

    def __init__(self, *nodes: Node[T, U]):
        """Create a new Graph by collecting all Inputs and Outputs from nodes."""
        self.inputs = []
        self.outputs = []

        for node in nodes:
            for incoming in node.incoming:
                if isinstance(incoming, Input):
                    self.inputs.append(incoming)
            for outgoing in node.outgoing:
                if isinstance(outgoing, Output):
                    self.outputs.append(outgoing)

    def copy(self) -> "Graph[T, U]":
        """Create a copy of the Graph by duplicating all Nodes and Edges.

        Node data and Edge contents are shallow-copied.
        """
        out = Graph()
        visited: dict[int, Node[T, U]] = {}

        for o in self.outputs:
            if o.from_ is None:
                raise ValueError("malformed graph: output comes from nowhere")

            if id(o.from_) in visited:
                n = visited[id(o.from_)]
            else:
                n = Node(o.from_.data)
                visited[id(o.from_)] = n

            out.outputs.extend(n.add_outputs(o.content))
            for e in o.from_.incoming:
                self._copy(out, e, n, visited)

        return out

    @staticmethod
    def _copy(
        out: "Graph[T, U]", e: Edge[T, U], to_node: Node[T, U], visited: dict[int, Node[T, U]]
    ):
        # e belongs to self; to_node is for the new graph

        if e.from_ is None:
            # input
            out.inputs.extend(to_node.add_inputs(e.content))
            return

        if id(e.from_) in visited:
            new_node = visited[id(e.from_)]
        else:
            # construct the new node
            new_node = Node(e.from_.data)
            visited[id(e.from_)] = new_node

        # node is constructed, just need to connect it
        new_node.new_edge_to(to_node, e.content)

        # continue up to the incoming edges
        for up in e.from_.incoming:
            Graph._copy(out, up, new_node, visited)

    def cut_input_layer(self) -> "Graph[T, U]":
        """Remove IN-PLACE all nodes that consume this graph's inputs, and make those nodes' outputs
        the inputs instead.

        For convenience, self is returned to allow for chaining.
        """
        old_inputs = self.inputs
        self.inputs = []

        removed: set[Node[T, U]] = set()

        for oi in old_inputs:
            dead_node = oi.to
            if dead_node is None:
                raise ValueError("malformed graph: unconsumed input")

            if dead_node in removed:
                continue

            for new_in in dead_node.outgoing:
                if new_in.to is None:
                    # discard this output
                    self.outputs.remove(cast(Output, new_in))
                    continue
                if new_in.to in removed:
                    # we're removing this node too, so discard this internal edge
                    continue

                # this edge becomes an input
                new_in.to.incoming.remove(new_in)
                self.inputs.extend(new_in.to.add_inputs(new_in.content))

            removed.add(dead_node)

        return self

    def cut_output_layer(self) -> "Graph[T, U]":
        """Remove IN-PLACE all nodes that produce this graph's outputs, and make the inputs to those
        nodes the outputs instead.

        For convenience, self is returned to allow for chaining.
        """
        old_outputs = self.outputs
        self.outputs = []

        removed: set[Node[T, U]] = set()

        for oo in old_outputs:
            dead_node = oo.from_
            if dead_node is None:
                raise ValueError("malformed graph: output comes from nowhere")

            if dead_node in removed:
                continue

            for new_out in dead_node.incoming:
                if new_out.from_ is None:
                    # discard this input
                    self.inputs.remove(cast(Input, new_out))
                    continue
                if new_out.from_ in removed:
                    # we're removing this node too, so discard this internal edge
                    continue

                # this edge becomes an output
                new_out.from_.outgoing.remove(new_out)
                self.outputs.extend(new_out.from_.add_outputs(new_out.content))

            removed.add(dead_node)

        return self

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

    def __bool__(self) -> bool:
        return bool(self.inputs) or bool(self.outputs)
