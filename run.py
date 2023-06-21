# ruff: noqa: T201
# This script needs to print.

import argparse
import asyncio
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
from langchain.embeddings import HuggingFaceEmbeddings
from language_tool_python import LanguageTool

import recipenlg
from evaluation.eval import evaluation, hallucination
from systems import FewShot, Model, System, ZeroShot
from workers import spread_gather


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


async def generate_and_evaluate(model: System, recipe: dict[str, Any], lt: LanguageTool):
    """Generate a recipe with the model, and then evaluate."""
    title = recipe["title"][0]

    logger = make_logger(str(recipe["id"][0]))

    res = await model.agenerate(title, logger)
    logger.debug(f"got {len(res)} generations")

    scores = defaultdict(list)
    scores["hallucination"].append(hallucination(res, logger))
    for completion in res:
        # make sure the recipe is correctly generated
        try:
            recipenlg.parse_recipe(completion)
        except ValueError:
            logger.warning(f"malformed recipe: {completion}")
            continue

        try:
            evals = await evaluation(completion, recipe, lt, logger)
        except Exception:
            logger.exception("exception during evaluation", exc_info=True)
            break
        for metric in evals:
            scores[metric].append(evals[metric])

    score_print = "\n".join(f"  {metric}: {scores[metric]}" for metric in scores)
    logger.debug("final scores:\n" + score_print)

    return scores, recipe["id"][0]


async def evaluate(model: System, data: Dataset, n_workers: int = 10):
    """Evaluate the system with the given recipe data."""
    n_workers = max(n_workers, len(data))
    with LanguageTool("en_US") as lt:
        results = await spread_gather(
            lambda recipe: generate_and_evaluate(model, recipe, lt),
            data.iter(1),
            n_workers,
            len(data),
        )

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
    for metric, score in scores.items():
        avg = np.round(np.mean(score), 3)
        print(f"{metric}: {avg}")

    if broken:
        print("\nno evaluation results for these IDs:", " ".join(str(i) for i in broken))


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
        "-s",
        "--system",
        type=str,
        default="ZeroShot",
        help="system to generate recipes",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="openai-gpt-3.5-turbo-0613",
        help="full name of service & model to use",
    )
    parser.add_argument("-n", type=int, default=sys.maxsize, help="number of samples to use")
    parser.add_argument(
        "--workers", type=int, default=10, help="number of concurrent requests to make to the LLM"
    )

    few_shot_options = parser.add_argument_group("FewShot")
    few_shot_options.add_argument(
        "--few-shot-k",
        type=int,
        default=3,
        help="maximum number of examples to provide (fewer are provided if they don't fit)",
    )
    few_shot_options.add_argument(
        "--few-shot-embed-model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="HuggingFace model to use for embeddings",
    )
    few_shot_options.add_argument(
        "--few-shot-embed-n",
        type=int,
        default=sys.maxsize,
        help=(
            "number of samples to use in the vector store. This is ignored when the vectors/ "
            "directory exists"
        ),
    )
    few_shot_options.add_argument(
        "--few-shot-embed-gpu", action="store_true", help="compute embeddings on the GPU"
    )

    args = parser.parse_args()

    model = Model.from_full_name(args.model, n=3)

    print("creating system...", file=sys.stderr)
    system = args.system.lower()
    if system == "zeroshot":
        system = ZeroShot(model)
    elif system == "fewshot":
        ds = recipenlg.load("train", args.data_dir)
        embedder = HuggingFaceEmbeddings(
            model_name=args.few_shot_embed_model,
            encode_kwargs=None if args.few_shot_embed_gpu else {"device": "cpu"},
        )
        system = FewShot(model, args.few_shot_k, ds, embedder, args.few_shot_embed_n)
    else:
        raise NotImplementedError(args.system)

    print("loading data...", file=sys.stderr)
    data = recipenlg.load("val", args.data_dir)
    n = min(args.n, len(data))
    data = data.select(np.arange(0, n))

    print("running evaluation...", file=sys.stderr)
    asyncio.run(evaluate(system, data, args.workers))
