import argparse
import asyncio

import numpy as np

import recipenlg
from evaluation.eval import coherence, consistency, quality, relevance
from recipenlg import format_recipe
from run import make_logger


def consistency_extraction(test_data):
    logger = make_logger("consistency")
    return [
        consistency(format_recipe(r["ingredients"], r["directions"]), logger) for r in test_data
    ]


def relevance_extraction(test_data):
    logger = make_logger("relevance")
    return [
        relevance(r["title"] + "\n" + format_recipe(r["ingredients"], r["directions"]), logger)
        for r in test_data
    ]


def coherence_extraction(test_data):
    logger = make_logger("coherence")
    return [
        coherence(r["title"] + "\n" + format_recipe(r["ingredients"], r["directions"]), logger)
        for r in test_data
    ]


def quality_extraction(test_data):
    logger = make_logger("quality")
    return [
        quality(r["title"] + "\n" + format_recipe(r["ingredients"], r["directions"]), logger)
        for r in test_data
    ]


async def worker(queue):
    while True:
        task = await queue.get()
        try:
            await task
        finally:
            queue.task_done()


async def main(data_dir: str = "./data", n_workers: int = 20, n: int = 3):
    data = recipenlg.load("val", data_dir)
    n = min(n, len(data))
    data = data.select(np.arange(0, 20))
    n_workers = min(n_workers, len(data))
    queue = asyncio.Queue()
    workers = []
    for _ in range(n_workers):
        workers.append(asyncio.create_task(worker(queue)))

    tasks = [quality_extraction(data)]
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
        default="./data",
        help="directory containing the RecipeNLG dataset",
    )
    parser.add_argument("-n", type=int, default=1, help="number of samples to use")
    parser.add_argument(
        "--workers", type=int, default=10, help="number of concurrent requests to make to the LLM"
    )
    args = parser.parse_args()
    asyncio.run(main(args.data_dir, args.workers, args.n))
