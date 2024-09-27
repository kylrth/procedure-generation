import unittest
from typing import cast

from graph import Graph, Node


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
    def test_copy(self):
        self.assertEqual(example_graph, example_graph.copy())

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
