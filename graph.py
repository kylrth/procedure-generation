from abc import abstractmethod
from enum import Enum, auto
from typing import Callable, Protocol, Self, Sequence, cast


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

    def new_edge_to(self, other: Self, edge: U) -> "Edge[T, U]":
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


class DFSAction(Enum):
    CONTINUE = auto()  # continue traversing down
    SKIP = auto()  # don't continue further down from the current edge
    QUIT = auto()  # stop traversal immediately


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
        """Create a new Graph by collecting all Inputs and Outputs from nodes.

        A ValueError is raised if not all Nodes and Inputs are reached by back-traversal from
        Outputs.
        """
        self.inputs = []
        self.outputs = []

        for node in nodes:
            for incoming in node.incoming:
                if isinstance(incoming, Input):
                    self.inputs.append(incoming)
                elif isinstance(incoming, Output):
                    raise TypeError("Output listed as incoming edge for Node")
            for outgoing in node.outgoing:
                if isinstance(outgoing, Output):
                    self.outputs.append(outgoing)
                elif isinstance(outgoing, Input):
                    raise TypeError("Input listed as outgoing edge for Node")

        # check that all Inputs and Nodes are reached by back-traversal
        self._check_reachable(nodes)

    def dfs(
        self,
        f: Callable[[Edge[T, U]], DFSAction],
        after: Callable[[Edge[T, U]], None] | None = None,
    ):
        """Call f on every Edge (including Inputs and Outputs) by doing depth-first search starting
        from the inputs. THIS IS NOT GUARANTEED TO DISCOVER ALL NODES.

        `after` is an optional function that will be called after having called f on all Edges below
        the current Edge.
        """
        if after is None:
            after = self._no_after

        visited: set[Node[T, U]] = set()
        self._dfs(self.inputs, f, after, visited)

    def back_dfs(
        self,
        f: Callable[[Edge[T, U]], DFSAction],
        after: Callable[[Edge[T, U]], None] | None = None,
    ):
        """Call f on every Edge (including Inputs and Outputs) by doing depth-first search starting
        from the outputs.

        `after` is an optional function that will be called after having called f on all Edges above
        the current Edge.
        """
        if after is None:
            after = self._no_after

        visited: set[Node[T, U]] = set()
        self._dfs(self.outputs, f, after, visited, backward=True)

    @staticmethod
    def _no_after(e: Edge[T, U]):
        pass

    @staticmethod
    def _dfs(
        es: Sequence[Edge[T, U]],
        f: Callable[[Edge[T, U]], DFSAction],
        after: Callable[[Edge[T, U]], None],
        visited: set[Node[T, U]],
        *,
        backward: bool = False,
    ) -> DFSAction:
        for e in es:
            action = f(e)
            if action == DFSAction.QUIT:
                return action
            if action == DFSAction.SKIP:
                continue

            nn = e.from_ if backward else e.to

            if nn is None:
                continue
            if nn in visited:
                continue

            visited.add(nn)  # mark as visited before continuing to avoid infinite loops
            action = Graph._dfs(
                nn.incoming if backward else nn.outgoing, f, after, visited, backward=backward
            )
            if action == DFSAction.QUIT:
                return action

            after(e)

        return DFSAction.CONTINUE

    def _check_reachable(self, nodes: Sequence[Node[T, U]]):
        reached_nodes: set[Node[T, U]] = set()
        reached_inputs: set[Input[T, U]] = set()

        def collect_nodes_inputs(e: Edge[T, U]) -> DFSAction:
            if e.from_ is None:
                if not isinstance(e, Input):
                    raise TypeError("Edge has an empty from_ field")
                reached_inputs.add(e)
            else:
                reached_nodes.add(e.from_)
            return DFSAction.CONTINUE

        self.back_dfs(collect_nodes_inputs)

        for node in nodes:
            if node not in reached_nodes:
                raise self.UnreachableError(node)
        for i in self.inputs:
            if i not in reached_inputs:
                raise self.UnreachableError(i)

    class UnreachableError(Exception):
        """Raised by the initializer when a Node or Input is unreachable by back-traversal from the
        outputs."""

        def __init__(self, missing: Node[T, U] | Input[T, U]):
            self.missing = missing

        def __str__(self):
            t = "node" if isinstance(self.missing, Node) else "input"
            data = self.missing.data if isinstance(self.missing, Node) else self.missing.content
            return f"{t} {data} was not reachable by back-traversal"

    def copy(self) -> Self:
        """Create a copy of the Graph by duplicating all Nodes and Edges.

        Node data and Edge contents are shallow-copied.
        """
        out = self.__class__()

        # keep newly-created nodes so we can refer to them when we come across their old
        # counterparts again
        visited: dict[Node[T, U], Node[T, U]] = {}

        def copy_edge(e: Edge[T, U]) -> DFSAction:
            # get our new copy of the to node
            if e.to is None:
                if e.from_ is None:
                    raise ValueError("malformed graph: completely empty Edge")
                # output
                new_to_node = None
            else:
                new_to_node = visited[e.to]

            if e.from_ is None:
                # input
                new_to_node = cast(Node[T, U], new_to_node)  # we already checked this above
                out.inputs.extend(new_to_node.add_inputs(e.content))
                return DFSAction.CONTINUE

            if e.from_ in visited:
                new_from_node = visited[e.from_]
            else:
                # construct the new node
                new_from_node = Node(e.from_.data)
                visited[e.from_] = new_from_node

            # node is constructed and stored in `visited`, just need to connect it
            if new_to_node is None:  # output
                out.outputs.extend(new_from_node.add_outputs(e.content))
            else:
                new_from_node.new_edge_to(new_to_node, e.content)
            return DFSAction.CONTINUE

        self.back_dfs(copy_edge)
        return out

    def topo_sort(self) -> list[Node[T, U]]:
        """Return a topological sort of the nodes, such that no node in the list depends on any
        later nodes in the list."""
        # we use an algorithm for topological sorting based on DFS
        # https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
        perm_mark: set[Node[T, U]] = set()
        temp_mark: set[Node[T, U]] = set()
        ordered: list[Node[T, U]] = []

        def topo_enter(e: Edge[T, U]) -> DFSAction:
            n = e.from_
            if n is None:
                return DFSAction.CONTINUE

            if n in perm_mark:
                return DFSAction.SKIP
            if n in temp_mark:
                raise self.DAGError

            temp_mark.add(n)
            return DFSAction.CONTINUE

        def topo_exit(e: Edge[T, U]):
            n = e.from_
            if n is None:
                return

            perm_mark.add(n)
            ordered.append(n)

        self.back_dfs(topo_enter, after=topo_exit)
        return ordered

    class DAGError(Exception):
        def __init__(self):
            super().__init__("graph has a loop")

    def cut_input_layer(self) -> Self:
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

    def cut_output_layer(self) -> Self:
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
        # produce any output. The latter case is an invalid state for a Graph to be in and will be
        # ignored.

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
