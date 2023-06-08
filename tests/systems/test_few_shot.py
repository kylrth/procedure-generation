import unittest

import numpy as np
from langchain.embeddings import FakeEmbeddings
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import FAISS

import recipenlg
import systems

from .mock_model import MockModel


example_data = [
    {
        "title": "Corn Casserole",
        "ingredients": [
            "1 small onion, chopped very fine",
            "1 can corn or 6 ears fresh corn cut off the cob",
            "2 level Tbsp. melted butter",
            "1 tsp. baking powder",
            "2/3 c. corn meal",
            "2 eggs",
            "1 tsp. salt",
            "1 1/2 c. milk",
        ],
        "directions": [
            "Heat milk, add butter and salt.",
            (
                "Cool slightly and add the corn meal and baking powder which have been sifted "
                "together, then the beaten eggs and chopped onion."
            ),
            "Stir well together.",
            (
                "Place the corn in a greased casserole and put the corn meal mixture over, mixing "
                "ever so slightly."
            ),
            "Mixture should be very thin.",
            "If necessary add a bit more milk.",
            "Bake, uncovered, in a moderate oven (350°) for 45 minutes or until nicely browned.",
        ],
    },
    {
        "title": "Orange Crinkles Cookies",
        "ingredients": [
            "1 box orange Duncan Hines cake mix",
            "1/2 c. oil",
            "2 eggs",
            "1/2 c. finely chopped nuts",
        ],
        "directions": ["Note:", "Different flavors cake mix can be used."],
    },
    {
        "title": "4 Fruit Wedding Punch Recipe",
        "ingredients": [
            "2 prt orange juice",
            "2 prt lemonade",
            "1 prt pineapple juice",
            "1 prt grapefruit juice (optional, but gives a nice tang)",
            "2 quart Lemonade from a mix, prepared",
            "1 can (12-ounce) frzn orange juice, reconstituted",
            '1/2 x "tall can" frzn pineapple juice, reconstituted',
            '1/2 x "tall can" grapefruit juice, reconstituted (optional)',
        ],
        "directions": [
            "This recipe contains no carbonation and no ice cream.",
            (
                "I have not tried it spiked, but you could probably add in rum to it (I'm not up "
                "on alcohol, but I believe which's supposed to be good in punch)."
            ),
            "This is our family's punch recipe-we serve it for all special occasions.",
            (
                "I will give what I think are the proportions first, then I'll give the exact "
                "amounts my mother-in-law uses for the 11 people in our extended family."
            ),
            (
                '*Sorry, I do not know the size of the "tall can", but it\'s the standard U.S. '
                "size which is larger than the 12-ounce.-can size."
            ),
            "Prepare lemonade.",
            (
                "Reconstitute orange juice, pineapple juice,and grapefruit juice, according to "
                "package directions, and in separate containers."
            ),
            "Add in lemonade to punch bowl.",
            "Add in all of the orange juice, 1/2",
            "(up to 3/4) of the pineapple juice, and 1/2 of the grapefruit juice.",
            "Slice thin orange wheels and float on top for a festive look.",
            "Cherries would also look nice.",
        ],
    },
]

few_shot_formatted = """
===BEGIN EXAMPLE===
Corn Casserole
Ingredients:
- 1 small onion, chopped very fine
- 1 can corn or 6 ears fresh corn cut off the cob
- 2 level Tbsp. melted butter
- 1 tsp. baking powder
- 2/3 c. corn meal
- 2 eggs
- 1 tsp. salt
- 1 1/2 c. milk
Instructions:
1. Heat milk, add butter and salt.
2. Cool slightly and add the corn meal and baking powder which have been sifted together, then the \
beaten eggs and chopped onion.
3. Stir well together.
4. Place the corn in a greased casserole and put the corn meal mixture over, mixing ever so \
slightly.
5. Mixture should be very thin.
6. If necessary add a bit more milk.
7. Bake, uncovered, in a moderate oven (350°) for 45 minutes or until nicely browned.
===END EXAMPLE===

===BEGIN EXAMPLE===
Orange Crinkles Cookies
Ingredients:
- 1 box orange Duncan Hines cake mix
- 1/2 c. oil
- 2 eggs
- 1/2 c. finely chopped nuts
Instructions:
1. Note:
2. Different flavors cake mix can be used.
===END EXAMPLE===
""".strip()


class TestFewShot(unittest.TestCase):
    def test_all(self):
        instructions = "Generate a recipe based on the title provided."
        titles = ["No Bake Nut Cookies"]
        data = [
            {
                "title": example["title"],
                "recipe": recipenlg.format_recipe(example["ingredients"], example["directions"]),
            }
            for example in example_data
        ]

        expected_prompts = [
            instructions + "\n\n" + few_shot_formatted + "\n\n" + title + "\n" for title in titles
        ]
        responses = [
            "<fake recipe for cookies>",
        ]
        model = MockModel(self, expected_prompts, responses)

        # seed for the fake embeddings so they're the same every time
        np.random.seed(27)  # noqa: NPY002  # FakeEmbeddings doesn't use Generator objects.
        selector = SemanticSimilarityExampleSelector.from_examples(
            data, FakeEmbeddings(size=10), FAISS, k=2
        )

        system = systems.FewShot(model, instructions, selector)

        self.assertEqual(responses[0], system.generate(titles[0])[0])


if __name__ == "__main__":
    unittest.main()
