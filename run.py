import argparse
import asyncio
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset

import recipenlg
from evaluation.eval import evaluation
from systems import Model, SystemInterface, ZeroShot


def make_logger(name: str) -> logging.Logger:
    """Create a new logger that writes to logs/{name}.log (and nowhere else)."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"

    logger = logging.getLogger(name)
    logger.propagate = False
    logger.addHandler(logging.FileHandler(log_file, "w"))
    logger.setLevel(logging.DEBUG)

    return logger


async def generate_and_evaluate(model: SystemInterface, recipe: dict[str, Any]):
    """Generate a recipe with the model, and then evaluate."""
    title = recipe["title"][0]
    ingredients = recipe["ingredients"][0]
    directions = recipe["directions"][0]
    recipe_text = recipenlg.format_recipe(ingredients, directions)

    logger = make_logger(str(recipe["id"][0]))

    res = await model.agenerate(title, logger)

    scores = defaultdict(list)
    for completion in res:
        # make sure the recipe is correctly generated
        try:
            recipenlg.parse_recipe(completion)
        except ValueError:
            logger.warning(f"malformed recipe: {completion}")
            continue

        evals = await evaluation(completion, recipe_text, logger)
        for metric in evals:
            scores[metric].append(evals[metric])

    score_print = "\n".join(f"  {metric}: {scores[metric]}" for metric in scores)
    logger.debug("final scores:\n" + score_print)

    return scores, recipe["id"][0]


async def worker(model, queue, results):
    while True:
        recipe = await queue.get()
        try:
            result = await generate_and_evaluate(model, recipe)
            results.append(result)
        finally:
            queue.task_done()


async def evaluate(model: SystemInterface, data: Dataset, n_workers: int = 10):
    """Evaluate the system with the given recipe data."""
    queue = asyncio.Queue()
    results = []

    n_workers = max(n_workers, len(data))
    workers = []
    for _ in range(n_workers):
        workers.append(asyncio.create_task(worker(model, queue, results)))

    for recipe in data.iter(1):
        await queue.put(recipe)

    await queue.join()
    for w in workers:
        w.cancel()

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

    if broken:
        print("The model returned bad responses for these titles:", " ".join(broken))


def main(model: str, data_dir: str = "./data", n: int = sys.maxsize, n_workers: int = 10):
    model = Model.from_full_name(model)
    system = ZeroShot(model)

    data = recipenlg.load("val", data_dir)
    n = min(n, len(data))
    data = data.select(np.arange(0, n))

    asyncio.run(evaluate(system, data, n_workers))


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
    parser.add_argument("-n", type=int, default=sys.maxsize, help="number of samples to use")
    parser.add_argument(
        "--workers", type=int, default=10, help="number of concurrent requests to make to the LLM"
    )

    args = parser.parse_args()

    main(args.model, args.data_dir, args.n, args.workers)
