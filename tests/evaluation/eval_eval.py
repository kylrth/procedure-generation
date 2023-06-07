import numpy as np
import recipenlg
from evaluation.eval import (
    ingredient_comparison,
    ingredient_consistency,
    ingredient_relevance,
    step_order,
    coherence,
)
from recipenlg import format_recipe, parse_recipe
from run import make_logger
from systems import Model, ZeroShot
import asyncio

data_dir = "C:\\Users\\mk_ya\\Desktop\\dataset\\dataset"


def ingredient_consistency_extraction(test_data):
    logger = make_logger("ingredient_consistency")
    tasks = [
        ingredient_consistency(format_recipe(r["ingredients"], r["directions"]), logger)
        for r in test_data
    ]
    return tasks


def ingredient_relevance_extraction(test_data):
    logger = make_logger("ingredient_relevance")
    tasks = [
        ingredient_relevance(
            r["title"] + "\n" + format_recipe(r["ingredients"], r["directions"]), logger
        )
        for r in test_data
    ]
    return tasks


def step_order_extraction(test_data):
    logger = make_logger("step_order")
    tasks = [step_order("Instructions:\n" + "\n".join(r["directions"]), logger) for r in test_data]
    return tasks


def coherence_extraction(test_data):
    logger = make_logger("coherence")
    tasks = [coherence(format_recipe(r["ingredients"], r["directions"]), logger) for r in test_data]
    return tasks


def ingredient_comparison_extraction(data):
    logger = make_logger("ingredient_comparison")
    model = Model.from_full_name("openai-gpt-3.5-turbo")
    system = ZeroShot(model, "Please generate a recipe")

    async def generate_and_compare(r):
        generated = await system.agenerate(r["title"])
        generated_ingredients, _ = parse_recipe(generated[0])
        await ingredient_comparison("\n".join(generated_ingredients), r["ingredients"], logger)

    tasks = [generate_and_compare(r) for r in data]
    return tasks


async def worker(queue):
    while True:
        task = await queue.get()
        try:
            await task
        finally:
            queue.task_done()


async def main(n_workers: int = 20):
    data = recipenlg.load("val", data_dir).select(np.arange(0, 3))
    n_workers = min(n_workers, len(data))
    queue = asyncio.Queue()
    workers = []
    for _ in range(n_workers):
        workers.append(asyncio.create_task(worker(queue)))

    tasks = [
        ingredient_consistency_extraction(data),
        ingredient_relevance_extraction(data),
        coherence_extraction(data),
        step_order_extraction(data),
        ingredient_comparison_extraction(data),
    ]
    for _task in tasks:
        for task in _task:
            await queue.put(task)
    await queue.join()
    for w in workers:
        w.cancel()


if __name__ == "__main__":
    asyncio.run(main())
