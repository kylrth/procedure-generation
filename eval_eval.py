import argparse
import asyncio
import sys

import numpy as np

import recipenlg
from evaluation.eval import (
    coherence,
    ingredient_comparison,
    consistency,
    relevance,
    structure,
)
from recipenlg import format_recipe, parse_recipe
from run import make_logger
from systems import Model, ZeroShot


def consistency_extraction(test_data):
    logger = make_logger("consistency")
    return [
        consistency(format_recipe(r["ingredients"], r["directions"]), logger)
        for r in test_data
    ]


def relevance_extraction(test_data):
    logger = make_logger("relevance")
    return [
        relevance(
            r["title"] + "\n" + format_recipe(r["ingredients"], r["directions"]), logger
        )
        for r in test_data
    ]


def structure_extraction(test_data):
    logger = make_logger("structure")
    return [structure(format_recipe(r["ingredients"], r["directions"]), logger) for r in test_data]


def coherence_extraction(test_data):
    logger = make_logger("coherence")
    return [coherence(r["title"]+'\n'+format_recipe(r["ingredients"], r["directions"]), logger) for r in test_data]


def ingredient_comparison_extraction(data):
    logger = make_logger("ingredient_comparison")
    model = Model.from_full_name("openai-gpt-3.5-turbo")
    system = ZeroShot(model, "Please generate a recipe")

    async def generate_and_compare(r):
        generated = await system.agenerate(r["title"])
        generated_ingredients, _ = parse_recipe(generated[0])
        await ingredient_comparison("\n".join(generated_ingredients), r["ingredients"], logger)

    return [generate_and_compare(r) for r in data]


async def worker(queue):
    while True:
        task = await queue.get()
        try:
            await task
        finally:
            queue.task_done()


async def main(data_dir: str = "./data", n_workers: int = 20, n: int = 3):
    data = recipenlg.load("val", data_dir).select(np.arange(0, n))
    n_workers = min(n_workers, len(data))
    queue = asyncio.Queue()
    workers = []
    for _ in range(n_workers):
        workers.append(asyncio.create_task(worker(queue)))

    tasks = [
        consistency_extraction(data),
        relevance_extraction(data),
        coherence_extraction(data),
        structure_extraction(data),
    ]
    for _task in tasks:
        for task in _task:
            await queue.put(task)
    await queue.join()
    for w in workers:
        w.cancel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data-dir",
        type=str,
        default="C:\\Users\\mk_ya\\Desktop\\dataset\\dataset",
        help="directory containing the RecipeNLG dataset",
    )
    parser.add_argument("-n", type=int, default=5, help="number of samples to use")
    parser.add_argument(
        "--workers", type=int, default=10, help="number of concurrent requests to make to the LLM"
    )
    args = parser.parse_args()
    asyncio.run(main(args.data_dir, args.workers, args.n))
