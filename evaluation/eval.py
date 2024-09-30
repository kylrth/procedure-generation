import argparse
import asyncio
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Awaitable

from dataset import LinearProcedure
from evaluation.api_overlap import ApiOverlap
from evaluation.edit_distance import EditDistance
from evaluation.heuristic import Heuristic
from evaluation.is_all_inp_used import AllInpUsed
from evaluation.number_comparison import NumberComparison
from evaluation.overall_score import OverallScore
from evaluation.rouge_score import ScoreROUGE
from evaluation.tfidf import TfIdf
from model import Model
from utils import spread_gather


async def evaluate_all(
    logger: logging.Logger,
    evals: dict[str, Heuristic],
    sample_id: int,
    gold: LinearProcedure,
    generated: list[str],
) -> dict[str, int | float]:
    """Evaluate a generated procedure by comparing with the gold procedure using various methods.

    The returned dictionary contains the evaluation result for each metric.
    """
    results: dict[str, int | float] = {"_id": sample_id}
    async_tasks = {}

    for name, heuristic_obj in evals.items():
        try:
            results[name] = heuristic_obj.evaluate(logger, gold, generated)
        except NotImplementedError:
            async_tasks[name] = heuristic_obj.aevaluate(logger, gold, generated)

    resp = await asyncio.gather(*async_tasks.values())
    # add async results to dict
    for name, result in zip(async_tasks.keys(), resp, strict=True):
        results[name] = result

    return results


async def write_results(w: csv.DictWriter, d: Awaitable[dict[str, int | float]]):
    w.writerow(await d)


def read_outputs_csv(path: Path) -> list[tuple[int, LinearProcedure, list[str]]]:
    out = []

    with path.open(newline="") as f:
        r = csv.DictReader(f)

        for row in r:
            out.append(
                (
                    int(row["question_id"]),
                    LinearProcedure(row["input"], row["output"], json.loads(row["gold_steps"])),
                    json.loads(row["completion"]),
                )
            )

    return out


def int_leq(v: int):
    """Custom type conversion function which validates that the int is less than or equal to v."""

    def validate(value):
        i = int(value)
        if i > v:
            raise argparse.ArgumentTypeError(f"{value} is not less than {v}")

        return i

    return validate


# constants
dataset_lcstep = "lcstep"
dataset_recipenlg = "recipenlg"
dataset_champ = "champ"


def get_evaluations(dataset: str, model: Model) -> dict[str, Heuristic]:
    evals = {}
    if dataset == dataset_recipenlg:
        evals["Ing_Used"] = AllInpUsed(model, dataset)
        evals["ROUGE"] = ScoreROUGE()
        evals["TfIdf"] = TfIdf(dataset)
        evals["Edit-Distance"] = EditDistance(model)
        evals["Num-Compare"] = NumberComparison(model)
        evals["Overall"] = OverallScore(model)
    elif dataset == dataset_lcstep:
        evals["Api-Overlap"] = ApiOverlap()
        evals["Inp-Used"] = AllInpUsed(model, dataset)
        evals["ROUGE"] = ScoreROUGE()
        evals["TfIdf"] = TfIdf(dataset)
        evals["Overall"] = OverallScore(model)
    elif dataset == dataset_champ:
        evals["Overall"] = OverallScore(model)

    return evals


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data-dir",
        type=str,
        default="./dataset",
        help="path to the dataset dir",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=dataset_lcstep,
        choices=[dataset_lcstep, dataset_recipenlg, dataset_champ],
        help="Dataset to run the system on",
    )

    parser.add_argument(
        "-s",
        "--system",
        type=str,
        default="RAG",
        help="system to perform generation",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="openai-gpt-3.5-turbo-0125",
        help="full name of service & model to use",
    )
    parser.add_argument(
        "-e",
        "--embedder",
        type=str,
        default="hf-all-mpnet-base-v2",
        help="full name of service & model to use for embeddings",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=sys.maxsize,
        help="ignored; allowed so scripts can make the dataset be 'recipenlg -n 100'",
    )
    parser.add_argument(
        "--workers", type=int, default=10, help="number of concurrent requests to make to the LLM"
    )

    args = parser.parse_args()
    args.system = args.system.lower()

    logger = logging.getLogger("main")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info(f"Running for: {args}")

    model = Model.from_full_name(args.model)

    logger.info("preparing the evaluations")
    evals = get_evaluations(args.dataset, model)

    in_path = Path("output") / args.system / args.dataset / args.embedder / "output.csv"
    out_path = in_path.parent / "eval_results.csv"

    logger.info("reading the outputs csv file")
    to_process = read_outputs_csv(in_path)

    async def _write_results(w: csv.DictWriter, d: Awaitable[dict[str, int | float]]):
        w.writerow(await d)

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, ["_id", *evals.keys()])
        w.writeheader()

        asyncio.run(
            spread_gather(
                lambda sample_data: _write_results(w, evaluate_all(logger, evals, *sample_data)),
                to_process,
                args.workers,
                len(to_process),
            )
        )
