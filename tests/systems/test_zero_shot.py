import unittest

import systems

from .mock_model import MockModel


class TestZeroShot(unittest.TestCase):
    def test_all(self):
        titles = ["No Bake Nut Cookies", "Spinach Dip", "Artichoke Casserole"]

        expected_prompts = [
            systems.ZeroShot.instructions + "\n\n" + title + "\n" for title in titles
        ]
        responses = [
            "one",
            "two",
            "three",
        ]
        model = MockModel(self, expected_prompts, responses)
        system = systems.ZeroShot(model)

        self.assertEqual(responses[0], system.generate(titles[0])[0])
        self.assertEqual(responses[1], system.generate(titles[1])[0])
        self.assertEqual(responses[2], system.generate(titles[2])[0])


if __name__ == "__main__":
    unittest.main()
