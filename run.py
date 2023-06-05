import argparse
import asyncio
from collections import defaultdict
import logging
import os
from typing import Any, Dict

from datasets import Dataset
import numpy as np

from evaluation.eval import evaluation
import recipenlg
from systems import Model, SystemInterface, ZeroShot


def make_logger(name: str) -> logging.Logger:
    """Create a new logger that writes to logs/{name}.log (and nowhere else)."""
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.addHandler(logging.FileHandler(f"logs/{name}.log", "w"))
    logger.setLevel(logging.DEBUG)

    return logger


async def generate_and_evaluate(model: SystemInterface, recipe: Dict[str, Any]):
    """Generate a recipe with the model, and then evaluate."""
    title = recipe["title"][0]
    ingredients = recipe["ingredients"][0]
    directions = recipe["directions"][0]

    logger = make_logger(str(recipe["id"][0]))

    res = await model.agenerate(title, logger)

    scores = defaultdict(list)
    for completion in res:
        # if recipe was incorrectly generated
        if "Instructions" not in completion or "Ingredients" not in completion:
            continue

        evals = await evaluation(
            completion, recipenlg.format_recipe(ingredients, directions), logger
        )
        for metric in evals:
            scores[metric].append(evals[metric])

    score_print = "\n".join(f"  {metric}: {scores[metric]}" for metric in scores)
    logger.debug("final scores:\n" + score_print)

    return scores, recipe["id"][0]


async def evaluate(model: SystemInterface, data: Dataset):
    """Evaluate the system with the given recipe data."""
    tasks = [generate_and_evaluate(model, recipe) for recipe in data.iter(1)]
    results = await asyncio.gather(*tasks)

    # collect results
    scores = defaultdict(list)
    broken = []
    for evals, rid in results:
        # if recipe had no correct completions
        if len(evals) == 0:
            broken.append(rid)
            continue
        # collect all scores of a metric
        for metric, values in evals.items():
            for v in values:
                scores[metric].append(v)

    # average the results
    scores_avg = defaultdict()
    for metric, score in scores.items():
        scores_avg[metric] = np.mean(score)

    print(scores_avg)
    print("The model returned bad responses for these titles:")
    print("\n".join(str(i) for i in broken))


def main(model: str, data_dir: str = "./data"):
    model = Model.from_full_name(model)
    system = ZeroShot(model, "Please generate a recipe")

    data = recipenlg.load("val", data_dir).select(np.arange(0, 3))

    asyncio.run(evaluate(system, data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data-dir",
        type=str,
        default="./data",
        help="directory containing the RecipeNLG dataset",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="openai-gpt-3.5-turbo",
        help="full name of service & model to use",
    )

    args = parser.parse_args()

    main(args.model, args.data_dir)
