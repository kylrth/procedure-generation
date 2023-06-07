"""Tools for the RecipeNLG data"""

import re

import numpy as np
from datasets import Dataset, load_dataset


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
        raise ValueError("split must be one of", splits.keys())

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


def _split_steps(steps: list[str]) -> list[str]:
    """Some recipes have steps that were not split correctly, so they ended up as a single step
    with several sentences. This splits those up."""
    if len(steps) != 1:
        return steps

    steps = steps[0].split(".")

    out = []
    for step in steps:
        stripped = step.strip()
        if stripped:
            out.append(stripped + ".")

    return out


def format_recipe(ingredients: list[str], directions: list[str]) -> str:
    """This is how we format recipes as text for models."""
    return "\n".join(
        (
            "Ingredients:",
            "\n".join("- " + ingredient for ingredient in ingredients),
            "Instructions:",
            "\n".join(f"{i+1}. {step}" for i, step in enumerate(directions)),
        )
    )


# matches "- ", "* ", or "1. ", "2. ", etc.
_markers = re.compile(r"([-*]|[0-9]+\.)\s*")


def parse_recipe(s: str) -> tuple[list[str], list[str]]:
    """Takes a recipe (title optional) and returns the list of ingredients and the list of
    instructions."""
    start = s.index("Ingredients:\n") + len("Ingredients:\n")
    try:
        end = s.index("\nInstructions:")
    except ValueError:
        end = s.index("\nDirections:")
    text = s[start:end]

    ingredients = []
    for t in text.split("\n"):
        if not t.strip():
            continue

        # remove any list markers
        ingredients.append(_markers.sub("", t))

    try:
        start = s.index("Instructions:\n") + len("Instructions:\n")
    except ValueError:
        start = s.index("Directions:\n") + len("Directions:\n")
    text = s[start:]

    instructions = []
    using_markers = False
    for t in text.split("\n"):
        if not t.strip():
            continue

        if _markers.match(t):
            using_markers = True
        elif using_markers:
            # If we've already seen instructions with markers, this line must not be part of the
            # instructions.
            break

        # remove any list markers
        instructions.append(_markers.sub("", t))

    return ingredients, instructions
