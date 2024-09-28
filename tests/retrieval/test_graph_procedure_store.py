import logging
import unittest

import numpy as np

from retrieval.embedder import Embedder
from retrieval.graph_procedure_store import GraphProcedureStore, Node, Recipe, Step
from utils.weaviate import NiceWeaviate


def _example_recipe() -> Recipe:
    step1 = Node(Step("preheat", "preheat the oven to 375°F", ["375°F"]))

    step2 = Node(Step("line tin", "line a muffin tin with paper liners or grease it"))
    step2.add_inputs("muffin tin", "paper muffin liners", "spray grease")

    step3 = Node(Step("mix", "mix dry ingredients"))
    step3.add_inputs(
        "2 cups all-purpose flour", "1 tbsp baking powder", "0.5 tsp salt", "0.75 cup sugar"
    )

    step4 = Node(Step("mix", "mix wet ingredients"))
    step4.add_inputs("1 cup milk", "2 large eggs", "0.5 cup melted butter", "1 tsp vanilla extract")

    step5 = Node(Step("combine", "combine wet and dry ingredients in a single bowl"))
    step3.new_edge_to(step5, "bowl of mixed dry ingredients")
    step4.new_edge_to(step5, "bowl of mixed wet ingredients")

    step6 = Node(Step("fill tin", "divide batter evenly among muffin cups, each about 2/3 full"))
    step2.new_edge_to(step6, "lined muffin tin")
    step5.new_edge_to(step6, "bowl of mixed ingredients")

    step7 = Node(Step("bake", "bake for 18-20 minutes", ["18", "20"]))
    step1.new_edge_to(step7, "warm oven")
    step6.new_edge_to(step7, "filled muffin tin")

    step8 = Node(
        Step("let cool", "let muffins cool in the pan for a few minutes", ["a few minutes"])
    )
    step7.new_edge_to(step8, "baked muffins in tin")

    step9 = Node(Step("remove muffins", "transfer muffins to a wire rack to cool completely"))
    step8.new_edge_to(step9, "cooled muffins in tin")
    step9.add_outputs("muffins")

    return Recipe(step1, step2, step3, step4, step5, step6, step7, step8, step9)


example_recipe = _example_recipe()


class TestEmbedder(Embedder):
    async def embed(self, text: list[str], *, is_query: bool = False) -> list[np.ndarray]:
        _ = is_query
        return [np.zeros(10) for _ in text]


class TestGraphProcedureStore(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.logger = logging.getLogger("test")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)

        self.client = NiceWeaviate()
        await self.client.__aenter__()

    async def asyncTearDown(self):
        await self.client.__aexit__(None, None, None)

    async def _new_test_store(self):
        store = GraphProcedureStore(self.client, TestEmbedder(), Recipe)
        await store.setup_collection(prefix=unittest.TestCase.id(self).split(".")[-1])
        return store

    async def test_add_get_one(self):
        store = await self._new_test_store()

        want_g = example_recipe
        want_v = np.random.default_rng().random(10)

        uuid = await store.add_graph(self.logger, want_g, want_v)

        got_g, got_v = await store.get_graph(uuid)

        self.assertEqual(want_g, got_g)
        np.testing.assert_almost_equal(want_v, got_v)

    async def test_search(self):
        store = await self._new_test_store()

        a = example_recipe
        b = a.copy().cut_input_layer().cut_input_layer()
        c = a.copy().cut_output_layer().cut_output_layer()

        av, bv, cv = np.identity(3)

        await store.add_graph(self.logger, a, av)
        await store.add_graph(self.logger, b, bv)
        await store.add_graph(self.logger, c, cv)

        self.assertEqual([a], await store.search_v(av, k=1))
        self.assertEqual([b, a, c], await store.search_v(np.array([0.1, 0.5, 0.0]), k=3))


if __name__ == "__main__":
    unittest.main()
