import argparse
import asyncio
import logging
import pickle
from pathlib import Path

from dataset import (
    CHAMP,
    Dataset,
    LCStep,
    LinearProcedure,
    RecipeNLG,
    Split,
    build_graph_with_retries,
)
from model import Model
from utils import spread_gather


async def convert_dataset_to_graphs(
    logger: logging.Logger, ds: Dataset, dataset: str, model: Model, workers: int
):
    procedures = ds.procedures(Split.TRAIN | Split.VAL | Split.TEST)
    root = Path("./dataset/graphs") / dataset
    root.mkdir(parents=True, exist_ok=True)

    async def _task(i: int, p: LinearProcedure):
        g = await build_graph_with_retries(logger, p, model, dataset)
        with (root / f"{i}.pkl").open("wb") as f:
            pickle.dump(g, f)

    await spread_gather(
        lambda item: _task(*item), enumerate(procedures), min(workers, 100), len(procedures)
    )


dataset_lcstep = "lcstep"
dataset_recipenlg = "recipenlg"
dataset_champ = "champ"

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
        "-m",
        "--model",
        type=str,
        default="openai-gpt-3.5-turbo-0125",
        help="full name of service & model to use",
    )
    parser.add_argument(
        "--workers", type=int, default=10, help="number of concurrent requests to make to the LLM"
    )

    args = parser.parse_args()

    logger = logging.getLogger("main")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    dataset_name = args.dataset.lower()
    if dataset_name == dataset_lcstep:
        ds = LCStep(args.data_dir)
    elif dataset_name == dataset_recipenlg:
        ds = RecipeNLG(args.data_dir, n=10000)
    elif dataset_name == dataset_champ:
        ds = CHAMP(args.data_dir)
    else:
        raise NotImplementedError(f"unrecognized dataset '{args.dataset}'")

    model = Model.from_full_name(args.model)

    asyncio.run(convert_dataset_to_graphs(logger, ds, args.dataset, model, args.workers))
