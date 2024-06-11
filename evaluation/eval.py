import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from dataset import Procedure
from evaluation.api_overlap import ApiOverlap
from evaluation.edit_distance import EditDistance
from evaluation.heuristic import Heuristic
from evaluation.is_all_inp_used import AllInpUsed
from evaluation.number_comparison import NumberComparison
from evaluation.overall_score import OverallScore
from evaluation.rouge_score import ScoreROUGE
from evaluation.tfidf import TfIdf
from systems import Model
from utils import spread_gather


async def evaluate_all(
    logger: logging.Logger,
    evals: dict[str, Heuristic],
    sample_id: int,
    gold: Procedure,
    generated: list[str],
):
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

    logger.debug("Collecting the heuristics scores for the example")
    resp = await asyncio.gather(*async_tasks.values())
    logger.info("Results of heuristics collected")
    # add async results to dict
    for name, result in zip(async_tasks.keys(), resp, strict=True):
        results[name] = result

    return results


def read_outputs_csv(args):
    output_dir = f"./output/{args.system}/{args.dataset}/{args.embedder}/output.csv"
    out_csv = pd.read_csv(
        output_dir, header=0, usecols=["question_id", "input", "output", "gold_steps", "completion"]
    )
    out_list = []
    for i in range(len(out_csv)):
        row = out_csv.to_numpy()[i]
        q_id = int(row[0])
        inputs = row[1]
        outputs = row[2]
        gold_steps = json.loads(row[3])
        gold = Procedure(input_=inputs, output=outputs, steps=gold_steps)
        gen_steps = json.loads(row[4])
        out_list.append((q_id, gold, gen_steps))

    return out_list


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


async def get_results(logger, out_list, dataset, model):
    evals = get_evaluations(dataset, model)

    out_evals = await spread_gather(
        lambda sample_data: evaluate_all(logger, evals, *sample_data[1]),
        enumerate(out_list),
        5,  # Num workers
        len(out_list),
    )
    return out_evals


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
        "-n", type=int, default=sys.maxsize, help="limit the number of samples to test"
    )
    parser.add_argument(
        "--workers", type=int, default=10, help="number of concurrent requests to make to the LLM"
    )
    parser.add_argument(
        "-k",
        type=int_leq(5),
        default=3,
        help=(
            "maximum number of examples to provide for FewShot and RAG (fewer are provided if they "
            "don't fit)"
        ),
    )

    args = parser.parse_args()
    logger = logging.getLogger("main")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.debug(f"Running for: {args}")

    cache_path = Path("cache")
    model = Model.from_full_name(args.model)
    logger.info("Initialized model")
    args.system = args.system.lower()

    # Read Outputs
    out_list = read_outputs_csv(args)
    logger.info("Read the outputs csv file")
    out_evals = asyncio.run(get_results(logger, out_list, args.dataset, model))

    to_record = pd.DataFrame(out_evals)
    to_record.to_csv(
        f"./output/{args.system}/{args.dataset}/{args.embedder}/eval_results.csv",
        header=True,
        index=False,
    )
