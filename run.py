# ruff: noqa: T201
# This script needs to print.
# ruff: noqa: I001, E402  # need to shut up before importing langchain

import os
import shutup

shutup.please()

import argparse
import asyncio
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import weaviate
from weaviate.embedded import EmbeddedOptions

import dataset
import retrieval
from evaluation.eval import evaluate_all
from dataset import Procedure
from systems import Model, System, aag, rag
from utils import log, spread_gather


async def generate_and_evaluate(
    model: System, item: tuple[int, Procedure], logger: log.ResultsLogger
) -> tuple[int, dict[str, Any]]:
    """Generate a procedure for this item with the model, and then evaluate it.

    Returns the item ID and the scores for each metric on each generated answer.
    """
    _id, p = item

    res = await model.agenerate(p.output, p._input)

    logger.result(
        log.Result(
            ID=_id,
            gold=p,
            prompt=res.prompt,
            completion=res.answer,
            retrieved_docs=res.retrieved_docs,
            context=res.context,
            model=res.model,
            input_tokens=res.input_tokens,
            output_tokens=res.output_tokens,
        )
    )

    try:
        evals = await evaluate_all(res.answer, p, logger)
        logger.evaluation(_id, evals)
    except Exception:
        logger.exception(_id, "exception during evaluation")
        evals = {}

    return _id, evals


async def evaluate(
    outdir: str | os.PathLike, model: System, data: list[Procedure], n_workers: int = 10
):
    """Evaluate the system with the given text generation dataset."""
    with log.ResultsLogger(outdir) as logger:
        results = await spread_gather(
            lambda prompt: generate_and_evaluate(model, prompt, logger),
            enumerate(data),
            n_workers,
            len(data),
        )

    # collect results
    scores = defaultdict(list)
    broken = []
    for _id, evals in results:
        # if there was an issue
        if len(evals) == 0:
            broken.append(_id)
            continue
        # collect all scores of a metric
        for metric, value in evals.items():
            scores[metric].append(value)

    # average the results
    for metric, score in scores.items():
        avg = np.round(np.mean(score), 3)
        print(f"{metric}: {avg}")

    if broken:
        print("\nno evaluation results for these IDs:", " ".join(str(i) for i in broken))


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
        default="openai-gpt-3.5-turbo-0613",
        help="full name of service & model to use",
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

    cache_path = Path("cache")

    dataset_name = args.dataset.lower()
    if dataset_name == dataset_lcstep:
        ds = dataset.LCStep(args.data_dir)
    elif dataset_name == dataset_recipenlg:
        ds = dataset.RecipeNLG(args.data_dir, n=10000)
    elif dataset_name == dataset_champ:
        ds = dataset.CHAMP(args.data_dir)
    else:
        raise NotImplementedError(f"unrecognized dataset '{args.dataset}'")

    logger.info("creating system...")
    model = Model.from_full_name(args.model)
    system_name = args.system.lower()
    with weaviate.WeaviateClient(
        embedded_options=EmbeddedOptions(
            persistence_data_path=str(Path("./cache/weaviate") / dataset_name),
            version="1.25.1",
            additional_env_vars={"AUTOSCHEMA_ENABLED": "false", "DISABLE_TELEMETRY": "true"},
        )
    ) as client:
        if system_name == "rag":
            # set up vector store with supporting docs + the train set of procedures
            logger.debug("RAG: collecting docs")
            docs = ds.docs(include_procedures=dataset.Split.TRAIN)
            logger.debug(f"RAG: collected {len(docs)} docs for vector store")

            client.collections.delete("Docs")
            docs_store = rag.setup_store(logger, client, name="Docs", desc="Supporting documents")

            logger.info("RAG: uploading docs to Weaviate collection")
            retrieval.populate(logger, system_name + "/" + dataset_name, docs_store, docs)

            system = rag.RAG(model, docs_store, args.k, args.dataset)
        elif system_name == "aag":
            # set up vector store for unchunked train procedures
            logger.debug("AAG: collecting train procedures")
            procedures = ds.procedures(dataset.Split.TRAIN)
            logger.debug(f"AAG: collected {len(procedures)} procedures for skill library")

            # TODO we're re-using the RAG vector store code for now, which does chunking and stuff
            client.collections.delete("Procedures")
            proc_store = rag.setup_store(logger, client, name="Procedures", desc="Skill library")

            logger.info("AAG: uploading docs to Weaviate collection")
            retrieval.populate(
                logger,
                system_name + "/" + dataset_name,
                proc_store,
                [p.to_doc() for p in procedures],
            )

            # TODO vector store for supporting docs

            system = aag.AAG(model, proc_store, args.k, args.dataset)
        else:
            raise NotImplementedError(args.system)

        outdir = Path("./output") / system_name / dataset_name

        # shorten eval set according to -n
        val_data = ds.procedures(dataset.Split.VAL)
        n = min(args.n, len(val_data))
        val_data = val_data[:n]
        logger.info(f"loaded {len(val_data)} eval examples")

        logger.info("starting evaluation...")
        asyncio.run(evaluate(outdir, system, val_data, min(args.workers, n)))
