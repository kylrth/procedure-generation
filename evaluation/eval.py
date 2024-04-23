import asyncio
from typing import Any

from dataset import Procedure
from utils import log


async def step_comparison(gold: Procedure, generated: str, logger: log.ResultsLogger) -> int:
    """Judge the generated steps by letting GPT-4 compare with the gold steps.

    Score is out of 10.
    """
    _ = gold
    _ = generated
    _ = logger

    return 10


async def evaluate_all(
    generated: str,
    gold: Procedure,
    logger: log.ResultsLogger,
) -> dict[str, Any]:
    """Evaluate a generated procedure by comparing with the gold procedure using various methods.

    The returned dictionary contains the evaluation result for each metric.
    """
    results = {}  # TODO add synchronous evals here

    async_tasks = {
        "compared": step_comparison(gold, generated, logger),
        # TODO add more asyncronous evals here
    }
    resp = await asyncio.gather(*async_tasks.values())

    # add async results to dict
    for name, result in zip(async_tasks.keys(), resp, strict=True):
        results[name] = result

    return results
