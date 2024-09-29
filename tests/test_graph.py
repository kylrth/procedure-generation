import unittest
from typing import cast

import pytest

from graph import DFSAction, Edge, Graph, Node


def _example_graph() -> Graph[str, str]:
    step1 = Node("preheat")

    step2 = Node("line tin")
    step2.add_inputs("muffin tin", "paper muffin liners", "spray grease")

    step3 = Node("mix dry")
    step3.add_inputs(
        "2 cups all-purpose flour", "1 tbsp baking powder", "0.5 tsp salt", "0.75 cup sugar"
    )

    step4 = Node("mix wet")
    step4.add_inputs("1 cup milk", "2 large eggs", "0.5 cup melted butter", "1 tsp vanilla extract")

    step5 = Node("combine")
    step3.new_edge_to(step5, "bowl of mixed dry ingredients")
    step4.new_edge_to(step5, "bowl of mixed wet ingredients")

    step6 = Node("fill tin")
    step2.new_edge_to(step6, "lined muffin tin")
    step5.new_edge_to(step6, "bowl of mixed ingredients")

    step7 = Node("bake")
    step1.new_edge_to(step7, "warm oven")
    step6.new_edge_to(step7, "filled muffin tin")

    step8 = Node("let cool")
    step7.new_edge_to(step8, "baked muffins in tin")

    step9 = Node("remove muffins")
    step8.new_edge_to(step9, "cooled muffins in tin")
    step9.add_outputs("muffins")

    return Graph(step1, step2, step3, step4, step5, step6, step7, step8, step9)


example_graph = _example_graph()


class TestGraph(unittest.TestCase):
    def test_init(self):
        a, b, c, d = (Node(i) for i in range(1, 5))
        a.add_inputs(5)
        a.new_edge_to(b, 6)
        b.new_edge_to(c, 7)

        # no outputs yet
        with pytest.raises(
            Graph.UnreachableError, match="node 1 was not reachable by back-traversal"
        ):
            Graph(a, b, c, d)

        c.add_outputs(8)
        with pytest.raises(
            Graph.UnreachableError, match="node 4 was not reachable by back-traversal"
        ):
            Graph(a, b, c, d)

        c.outgoing.clear()
        c.new_edge_to(d, 9)
        d.add_outputs(10)
        Graph(a, b, c, d)  # success

        d.outgoing.clear()
        d.new_edge_to(a, 11)  # a loop with no outputs
        with pytest.raises(
            Graph.UnreachableError, match="node 1 was not reachable by back-traversal"
        ):
            Graph(a, b, c, d)

        c.add_outputs(12)  # a loop with an output
        Graph(a, b, c, d)  # success

        e = Node(13)
        e.add_inputs(14)
        with pytest.raises(
            Graph.UnreachableError, match="node 13 was not reachable by back-traversal"
        ):
            Graph(a, b, c, d, e)

        e.add_outputs(15)
        Graph(a, b, c, d, e)  # success with two entirely separate subgraphs

    def test_copy(self):
        self.assertEqual(example_graph, example_graph.copy())

    def test_topo_sort(self):
        # test with example graph
        nodes = example_graph.topo_sort()

        def check_topo(e: Edge) -> DFSAction:
            if e.from_ is None or e.to is None:
                return DFSAction.CONTINUE

            if nodes.index(e.from_) >= nodes.index(e.to):
                self.fail(f"node {e.from_.data} came after {e.to.data}")

            return DFSAction.CONTINUE

        self.assertEqual(9, len(nodes))
        example_graph.back_dfs(check_topo)

        # graph with two outputs
        a, b, c = Node(1), Node(2), Node(3)
        a.new_edge_to(b, 4)
        b.new_edge_to(c, 5)
        c.add_outputs(6, 7)
        g = Graph(a, b, c)
        nodes = g.topo_sort()
        self.assertEqual(3, len(nodes))
        g.back_dfs(check_topo)

        # should fail with loop
        a, b, c = Node(1), Node(2), Node(3)
        a.new_edge_to(b, 4)
        b.new_edge_to(c, 5)
        c.new_edge_to(a, 6)
        c.add_outputs(7)
        g = Graph(a, b, c)
        with pytest.raises(Graph.DAGError):
            g.topo_sort()

    def test_eq(self):
        self.assertNotEqual(example_graph, Graph())
        self.assertNotEqual(example_graph, example_graph.copy().cut_input_layer())
        self.assertNotEqual(example_graph, example_graph.copy().cut_output_layer())

    def test_cut_input_layer(self):
        cut = example_graph.copy().cut_input_layer()  # remove steps 2, 3, 4
        self.assertEqual(
            ["lined muffin tin", "bowl of mixed dry ingredients", "bowl of mixed wet ingredients"],
            [i.content for i in cut.inputs],
        )
        self.assertEqual(
            {"fill tin", "combine"}, {cast(Node[str, str], i.to).data for i in cut.inputs}
        )

        cut.cut_input_layer()  # remove steps 5 and 6

        self.assertEqual(
            ["filled muffin tin"],
            [i.content for i in cut.inputs],
        )
        self.assertEqual({"bake"}, {cast(Node[str, str], i.to).data for i in cut.inputs})

        cut.cut_input_layer()  # remove step 7 (indirectly removing 1)

        self.assertEqual(
            ["baked muffins in tin"],
            [i.content for i in cut.inputs],
        )
        self.assertEqual({"let cool"}, {cast(Node[str, str], i.to).data for i in cut.inputs})

        cut.cut_input_layer()  # remove step 8
        cut.cut_input_layer()  # remove step 9

        self.assertEqual([], cut.inputs)
        self.assertEqual([], cut.outputs)

    def test_cut_output_layer(self):
        cut = example_graph.copy().cut_output_layer()  # remove step 9
        self.assertEqual(["cooled muffins in tin"], [o.content for o in cut.outputs])
        self.assertEqual({"let cool"}, {cast(Node[str, str], o.from_).data for o in cut.outputs})

        cut.cut_output_layer()  # remove step 8
        cut.cut_output_layer()  # remove step 7

        self.assertEqual(["warm oven", "filled muffin tin"], [o.content for o in cut.outputs])
        self.assertEqual(
            {"preheat", "fill tin"}, {cast(Node[str, str], o.from_).data for o in cut.outputs}
        )

        cut.cut_output_layer()  # remove steps 1 and 6
        cut.cut_output_layer()  # remove steps 2 and 5
        cut.cut_output_layer()  # remove steps 3 and 4

        self.assertEqual([], cut.inputs)
        self.assertEqual([], cut.outputs)


if __name__ == "__main__":
    unittest.main()
