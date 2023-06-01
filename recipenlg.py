"""Tools for the RecipeNLG data"""

from typing import List, Tuple

from datasets import Dataset, load_dataset
import numpy as np

# deterministic shuffle not tied to any seed value anywhere else
rng = np.random.default_rng(27)


def load(split: str = "train", data_dir: str = "./data") -> Dataset:
    """Load the specified split of the RecipeNLG dataset. This dataset does not have set splits, so
    we specify them with a separate random number generator seeded with a constant value."""
    split = split.lower()

    splits = {
        "train": 0.0,
        "val": 0.9,
        "test": 0.95,
    }

    if split not in splits:
        raise ValueError(f"split must be one of {splits.keys()}")

    ds = load_dataset("recipe_nlg", data_dir=data_dir)["train"]
    ds = ds.shuffle(generator=rng)

    begin_val = int(len(ds) * splits["val"])
    begin_test = int(len(ds) * splits["test"])

    if split == "train":
        return _preprocess(ds.select(np.arange(0, begin_val)))
    if split == "val":
        return _preprocess(ds.select(np.arange(begin_val, begin_test)))

    return _preprocess(ds.select(np.arange(begin_test, len(ds))))


def _preprocess(ds: Dataset) -> Dataset:
    ds = ds.map(
        lambda x: {
            "directions": _split_steps(x["directions"]),
        }
    )

    return ds


def _split_steps(steps: List[str]) -> List[str]:
    """Some recipes have steps that were not split correctly, so they ended up as a single step
    with several sentences. This splits those up."""
    if len(steps) != 1:
        return steps

    steps = steps[0].split(".")

    out = []
    for step in steps:
        step = step.strip()
        if step:
            out.append(step + ".")

    return out


def format_recipe(ingredients: List[str], directions: List[str]) -> str:
    """This is how we format recipes as text for models."""
    return "\n".join(
        (
            "Ingredients:",
            "\n".join("- " + ingredient for ingredient in ingredients),
            "Instructions:",
            "\n".join(f"{i+1}. {step}" for i, step in enumerate(directions)),
        )
    )


def parse_recipe(s: str) -> Tuple[List[str], List[str]]:
    """Takes a recipe (title optional) and returns the list of ingredients and the list of
    instructions."""
    start, end = s.index("Ingredients:\n") + len("Ingredients:\n"), s.index("\nInstructions:")
    ingredients = s[start:end]
    ingredients = [ingredient.strip("- ") for ingredient in ingredients.split("\n")]

    start = s.index("Instructions:\n") + len("Instructions:\n")
    instructions = s[start:]
    instructions = [instruction.split(". ", 1)[1] for instruction in instructions.split("\n")]

    return ingredients, instructions
