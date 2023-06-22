# ruff: noqa: E501

import unittest

from datasets import Dataset

import recipenlg


# These are the first three examples from the actual recipenlg dataset, plus an extra one to make
# sure we handle the case where there really is only one step.
testset = Dataset.from_list(
    [
        {
            "id": 0,
            "title": "No-Bake Nut Cookies",
            "ingredients": [
                "1 c. firmly packed brown sugar",
                "1/2 c. evaporated milk",
                "1/2 tsp. vanilla",
                "1/2 c. broken nuts (pecans)",
                "2 Tbsp. butter or margarine",
                "3 1/2 c. bite size shredded rice biscuits",
            ],
            "directions": [
                (
                    "In a heavy 2-quart saucepan, mix brown sugar, nuts, evaporated milk and "
                    "butter or margarine."
                ),
                "Stir over medium heat until mixture bubbles all over top.",
                "Boil and stir 5 minutes more. Take off heat.",
                "Stir in vanilla and cereal; mix well.",
                "Using 2 teaspoons, drop and shape into 30 clusters on wax paper.",
                "Let stand until firm, about 30 minutes.",
            ],
            "link": "www.cookbooks.com/Recipe-Details.aspx?id=44874",
            "source": 0,
            "ner": [
                "brown sugar",
                "milk",
                "vanilla",
                "nuts",
                "butter",
                "bite size shredded rice biscuits",
            ],
        },
        {
            "id": 1,
            "title": "Jewell Ball'S Chicken",
            "ingredients": [
                "1 small jar chipped beef, cut up",
                "4 boned chicken breasts",
                "1 can cream of mushroom soup",
                "1 carton sour cream",
            ],
            "directions": [
                "Place chipped beef on bottom of baking dish.",
                "Place chicken on top of beef.",
                (
                    "Mix soup and cream together; pour over chicken. Bake, uncovered, at 275° for "
                    "3 hours."
                ),
            ],
            "link": "www.cookbooks.com/Recipe-Details.aspx?id=699419",
            "source": 0,
            "ner": ["beef", "chicken breasts", "cream of mushroom soup", "sour cream"],
        },
        {
            "id": 2,
            "title": "Creamy Corn",
            "ingredients": [
                "2 (16 oz.) pkg. frozen corn",
                "1 (8 oz.) pkg. cream cheese, cubed",
                "1/3 c. butter, cubed",
                "1/2 tsp. garlic powder",
                "1/2 tsp. salt",
                "1/4 tsp. pepper",
            ],
            "directions": [
                (
                    "In a slow cooker, combine all ingredients. Cover and cook on low for 4 hours "
                    "or until heated through and cheese is melted. Stir well before serving. "
                    "Yields 6 servings."
                )
            ],
            "link": "www.cookbooks.com/Recipe-Details.aspx?id=10570",
            "source": 0,
            "ner": ["frozen corn", "cream cheese", "butter", "garlic powder", "salt", "pepper"],
        },
        {
            "id": 3,
            "title": "Fake Recipe",
            "ingredients": [
                "nothing",
                "more nothing",
            ],
            "directions": ["Do whatever you want, there's no manual for life."],
            "link": "https://example.com",
            "source": 2,
            "ner": ["nothing"],
        },
    ]
)
testset = testset.map(
    lambda x: {
        "formatted": (x["title"])
        + "\n\n"
        + recipenlg.format_recipe(x["ingredients"], x["directions"]),
    },
)

# This is what the recipe for id 2 should look like after formatting the directions.
id2_formatted = {
    "id": 2,
    "title": "Creamy Corn",
    "ingredients": [
        "2 (16 oz.) pkg. frozen corn",
        "1 (8 oz.) pkg. cream cheese, cubed",
        "1/3 c. butter, cubed",
        "1/2 tsp. garlic powder",
        "1/2 tsp. salt",
        "1/4 tsp. pepper",
    ],
    "directions": [
        "In a slow cooker, combine all ingredients.",
        "Cover and cook on low for 4 hours or until heated through and cheese is melted.",
        "Stir well before serving.",
        "Yields 6 servings.",
    ],
    "formatted": """Creamy Corn

Ingredients:
- 2 (16 oz.) pkg. frozen corn
- 1 (8 oz.) pkg. cream cheese, cubed
- 1/3 c. butter, cubed
- 1/2 tsp. garlic powder
- 1/2 tsp. salt
- 1/4 tsp. pepper

Instructions:
1. In a slow cooker, combine all ingredients.
2. Cover and cook on low for 4 hours or until heated through and cheese is melted.
3. Stir well before serving.
4. Yields 6 servings.
""".strip(),
    "link": "www.cookbooks.com/Recipe-Details.aspx?id=10570",
    "source": 0,
    "ner": ["frozen corn", "cream cheese", "butter", "garlic powder", "salt", "pepper"],
}


class TestPreprocess(unittest.TestCase):
    maxDiff = None

    def test_basic(self):
        got = recipenlg._preprocess(testset)

        self.assertEqual(testset[0], got[0])
        self.assertEqual(testset[1], got[1])
        self.assertEqual(id2_formatted, got[2])
        self.assertEqual(testset[3], got[3])


format_variations = {
    "basic": """Creamy Corn

Ingredients:
- 2 (16 oz.) pkg. frozen corn
- 1 (8 oz.) pkg. cream cheese, cubed
- 1/3 c. butter, cubed
- 1/2 tsp. garlic powder
- 1/2 tsp. salt
- 1/4 tsp. pepper

Instructions:
1. In a slow cooker, combine all ingredients.
2. Cover and cook on low for 4 hours or until heated through and cheese is melted.
3. Stir well before serving.
4. Yields 6 servings.

Enjoy this recipe for creamy corn!
""",
    "directions_bullets": """Ingredients:
- 2 (16 oz.) pkg. frozen corn
- 1 (8 oz.) pkg. cream cheese, cubed
- 1/3 c. butter, cubed
- 1/2 tsp. garlic powder
- 1/2 tsp. salt
- 1/4 tsp. pepper

Directions:
- In a slow cooker, combine all ingredients.
- Cover and cook on low for 4 hours or until heated through and cheese is melted.
- Stir well before serving.
- Yields 6 servings.
""",
    "alt_numbering": """Ingredients:
* 2 (16 oz.) pkg. frozen corn
* 1 (8 oz.) pkg. cream cheese, cubed
* 1/3 c. butter, cubed
* 1/2 tsp. garlic powder
* 1/2 tsp. salt
* 1/4 tsp. pepper

Instructions:
1. In a slow cooker, combine all ingredients.
1. Cover and cook on low for 4 hours or until heated through and cheese is melted.
10. Stir well before serving.
1. Yields 6 servings.
""",
    "no_numbering": """Ingredients:
2 (16 oz.) pkg. frozen corn
1 (8 oz.) pkg. cream cheese, cubed
1/3 c. butter, cubed
1/2 tsp. garlic powder
1/2 tsp. salt
1/4 tsp. pepper

Instructions:
In a slow cooker, combine all ingredients.
Cover and cook on low for 4 hours or until heated through and cheese is melted.
Stir well before serving.
Yields 6 servings.
""",
    "spread_out": """Creamy Corn

Ingredients:
- 2 (16 oz.) pkg. frozen corn
- 1 (8 oz.) pkg. cream cheese, cubed
- 1/3 c. butter, cubed
- 1/2 tsp. garlic powder
- 1/2 tsp. salt
- 1/4 tsp. pepper

Instructions:

1. In a slow cooker, combine all ingredients.

2. Cover and cook on low for 4 hours or until heated through and cheese is melted.

3. Stir well before serving.

4. Yields 6 servings.
""",
}


class TestRecipeFormat(unittest.TestCase):
    maxDiff = None

    def test_parse_variations(self):
        for name, variation in format_variations.items():
            with self.subTest(name):
                ingredients, instructions = recipenlg.parse_recipe(variation)

                self.assertEqual(id2_formatted["ingredients"], ingredients)
                self.assertEqual(id2_formatted["directions"], instructions)

    def test_reversible(self):
        s = recipenlg.format_recipe(id2_formatted["ingredients"], id2_formatted["directions"])
        got_i, got_d = recipenlg.parse_recipe(s)

        self.assertEqual(id2_formatted["ingredients"], got_i)
        self.assertEqual(id2_formatted["directions"], got_d)

    def test_internal_numbers_hyphens(self):
        # make sure internal numbers and hyphens are handled correctly
        original = """Ingredients:
- 1 can (15.25 oz) whole kernel corn, drained
- 1 can (14.75 oz) cream-style corn
- 1 cup cornmeal
- 1/2 cup all-purpose flour
- 1/4 cup granulated sugar
- 1/2 teaspoon baking powder
- 1/2 teaspoon salt
- 1/4 teaspoon black pepper
- 1/2 cup melted butter
- 1 cup sour cream
- 2 large eggs, beaten
- 1 cup shredded cheddar cheese

Instructions:
1. Preheat the oven to 375°F (190°C). Grease a 9x9-inch baking dish.
2. In a large bowl, combine the whole kernel corn, cream-style corn, cornmeal, flour, sugar, baking powder, salt, and black pepper. Mix well.
3. Add the melted butter, sour cream, beaten eggs, and shredded cheddar cheese to the corn mixture. Stir until all ingredients are well combined.
4. Pour the mixture into the greased baking dish, spreading it evenly.
5. Bake in the preheated oven for 45-50 minutes, or until the top is golden brown and the casserole is set in the center.
6. Remove from the oven and let it cool for a few minutes before serving.
7. Serve warm as a side dish or as a main course with a salad or vegetables. Enjoy!"""

        ing, ins = recipenlg.parse_recipe(original)
        got = recipenlg.format_recipe(ing, ins)

        self.assertEqual(original, got)


if __name__ == "__main__":
    unittest.main()
