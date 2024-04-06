# ruff: noqa: T201
# This script needs to print.
# ruff: noqa: I001, E402  # need to shut up before importing langchain

import shutup

shutup.please()

import argparse
import asyncio
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import torch

torch.cuda.is_available = lambda: False  # set this before importing retrieval which loads a model

import weaviate
from weaviate.embedded import EmbeddedOptions

import retrieval
from dataset import lcstep
from evaluation.eval import evaluate_all
from systems import Model, Result, System, aag, rag
from utils import data, log, spread_gather

if TYPE_CHECKING:
    from dataset.base import Procedure


def create_log_result(_id: int, source: str, res: Result, label: str) -> log.Result:
    return log.Result(
        ID=_id,
        source=source,
        query=res.query,
        label=label,
        prompt=res.prompt,
        completions=res.answers,
        retrieved_docs=res.retrieved_docs if res.retrieved_docs is not None else [],
        context=res.context if res.context is not None else "",
        model=res.model,
    )


async def generate_and_evaluate(
    model: System, item: dict[str, Any], logger: log.ResultsLogger
) -> tuple[int, dict[str, list[Any]]]:
    """Generate a procedure for this item with the model, and then evaluate it.

    Returns the item ID and the scores for each metric on each generated answer.
    """
    p: Procedure = item["procedure"]
    _id: int = item["id"]
    source: str = item["path"]
    ref: str = "\n".join(p.steps)

    res = await model.agenerate(p.output)
    logger.result(create_log_result(_id, source, res, ref))

    scores = defaultdict(list)
    for answer in res.answers:
        try:
            evals = await evaluate_all(answer, item, logger)
            logger.evaluation(_id, evals)
        except Exception:
            logger.exception(_id, "exception during evaluation")
            continue
        for metric in evals:
            scores[metric].append(evals[metric])

    return _id, scores


async def evaluate(model: System, data: list[dict[str, Any]], n_workers: int = 10):
    """Evaluate the system with the given text generation dataset."""
    with log.ResultsLogger("output.csv", "logs") as logger:
        results = await spread_gather(
            lambda prompt: generate_and_evaluate(model, prompt, logger),
            data,
            n_workers,
            len(data),
        )

    # collect results
    scores = defaultdict(list)
    broken = []
    for _id, evals in results:
        # if prompt had no correct completions
        if len(evals) == 0:
            broken.append(_id)
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


def int_leq(v: int):
    """Custom type conversion function which validates that the int is less than or equal to v."""

    def validate(value):
        i = int(value)
        if i > v:
            raise argparse.ArgumentTypeError(f"{value} is not less than {v}")

        return i

    return validate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data-dir",
        type=str,
        default="./dataset/LCStep/docs",
        help="directory containing the LCStep dataset",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="lcstep",
        choices=["lcstep", "recipe_nlg"],
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

    logger.info("loading data...")
    if args.dataset.lower() == "lcstep":
        lcstep_data = lcstep.load_formatted_docs(args.data_dir)
        train_data, val_data, _ = data.train_val_test(lcstep_data, val=0.1, test=0.2)
    else:
        raise NotImplementedError(f"unrecognized dataset '{args.dataset}'")

    # shorten eval set according to -n
    n = min(args.n, len(val_data))
    val_data = val_data[:n]
    logger.info(f"loaded {len(val_data)} eval examples")

    logger.info("creating system...")
    model = Model.from_full_name(args.model)
    system = args.system.lower()
    with weaviate.WeaviateClient(
        embedded_options=EmbeddedOptions(
            persistence_data_path="./cache/weaviate",
            version="1.24.6",
            additional_env_vars={"AUTOSCHEMA_ENABLED": "false", "DISABLE_TELEMETRY": "true"},
        )
    ) as client:
        if system == "rag":
            # set up vector store with training data, API refs concept docs
            logger.debug("RAG: collecting docs")

            # convert all procedures in training data to dicts with keys "title" and "contents"
            proc_docs = []
            for d in train_data:
                proc_docs.append(
                    {
                        "title": d["procedure"].output,
                        "contents": d["procedure"]._input + "\n\n" + " ".join(d["procedure"].steps),
                    }
                )

            api_ref = lcstep.load_api_ref(args.data_dir)
            concept_docs = lcstep.load_concept_docs(args.data_dir)

            docs = proc_docs + api_ref + concept_docs
            logger.debug(f"RAG: collected {len(docs)} docs for vector store")

            docs_store = rag.setup_store(
                logger, client, name="Docs", desc="Documentation for the LangChain Python library."
            )

            if len(docs_store) == 0:
                logger.info(f"uploading {len(docs)} chunks to Weaviate collection")
                retrieval.populate(logger, docs_store, docs)

            system = rag.RAG(model, docs_store, args.k)
        elif system == "aag":
            # set up API ref store
            api_ref = lcstep.load_api_ref(args.data_dir)
            api_ref = api_ref.rename_column("ref", "documentation")
            aag.setup_api_ref(logger, client, list(api_ref.iter(1)))

            # set up skill library from concept docs
            concept_docs = lcstep.load_concept_docs(args.data_dir)
            skills = asyncio.run(aag.build_concept_skills(model, concept_docs))
            aag.setup_skills(logger, client, skills)

            system = aag.AAG(model, client)
        else:
            raise NotImplementedError(args.system)

        logger.info("starting evaluation...")
        asyncio.run(evaluate(system, val_data, min(args.workers, n)))
