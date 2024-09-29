import unittest

from dataset.base import GraphProcedure, Step
from graph import Node


def _example_procedure() -> GraphProcedure:
    text = "preheat"
    step1 = Node(Step(text, text))

    text = "line tin"
    step2 = Node(Step(text, text))
    step2.add_inputs("muffin tin", "paper muffin liners", "spray grease")

    text = "mix dry"
    step3 = Node(Step(text, text))
    step3.add_inputs(
        "2 cups all-purpose flour", "1 tbsp baking powder", "0.5 tsp salt", "0.75 cup sugar"
    )

    text = "mix wet"
    step4 = Node(Step(text, text))
    step4.add_inputs("1 cup milk", "2 large eggs", "0.5 cup melted butter", "1 tsp vanilla extract")

    text = "combine"
    step5 = Node(Step(text, text))
    step3.new_edge_to(step5, "bowl of mixed dry ingredients")
    step4.new_edge_to(step5, "bowl of mixed wet ingredients")

    text = "fill tin"
    step6 = Node(Step(text, text))
    step2.new_edge_to(step6, "lined muffin tin")
    step5.new_edge_to(step6, "bowl of mixed ingredients")

    text = "bake"
    step7 = Node(Step(text, text))
    step1.new_edge_to(step7, "warm oven")
    step6.new_edge_to(step7, "filled muffin tin")

    text = "let cool"
    step8 = Node(Step(text, text))
    step7.new_edge_to(step8, "baked muffins in tin")

    text = "remove muffins"
    step9 = Node(Step(text, text))
    step8.new_edge_to(step9, "cooled muffins in tin")
    step9.add_outputs("muffins")

    return GraphProcedure(step1, step2, step3, step4, step5, step6, step7, step8, step9)


example_procedure = _example_procedure()


class TestGraphProcedure(unittest.TestCase):
    def test_str(self):
        got = str(example_procedure)

        want = """goal: muffins
inputs: muffin tin, paper muffin liners, spray grease, 2 cups all-purpose flour, 1 tbsp baking powder, 0.5 tsp salt, 0.75 cup sugar, 1 cup milk, 2 large eggs, 0.5 cup melted butter, 1 tsp vanilla extract
- preheat
- line tin
- mix dry
- mix wet
- combine
- fill tin
- bake
- let cool
- remove muffins"""  # noqa: E501

        self.assertEqual(want, got)


if __name__ == "__main__":
    unittest.main()
