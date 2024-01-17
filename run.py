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
from typing import Any

import numpy as np
import weaviate
from datasets import Dataset, concatenate_datasets
from langchain.text_splitter import RecursiveCharacterTextSplitter

import lcstep
from evaluation.eval import evaluate_all
from systems import Model, Result, System, aag, rag
from utils import log, spread_gather


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
    goal: str = item["goal"][0]
    _id: int = item["id"][0]
    source: str = item["path"][0]
    ref: str = item["ref"][0]

    res = await model.agenerate(goal)
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


async def evaluate(model: System, data: Dataset, n_workers: int = 10):
    """Evaluate the system with the given text generation dataset."""
    with log.ResultsLogger("output.csv", "logs") as logger:
        results = await spread_gather(
            lambda prompt: generate_and_evaluate(model, prompt, logger),
            data.iter(1),
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
        default="./dataset/docs",
        help="directory containing the LCStep dataset",
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
    parser.add_argument("-n", type=int, default=sys.maxsize, help="number of samples to use")
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
    data = lcstep.load_formatted_docs(args.data_dir)

    # shorten dataset according to -n
    n = min(args.n, len(data))
    data = data.select(np.arange(0, n))

    logger.info("creating system...")
    model = Model.from_full_name(args.model)
    system = args.system.lower()
    if system == "rag":
        store = weaviate.connect_to_local()

        # set up vector store with API refs and concept docs
        api_ref = lcstep.load_api_ref(args.data_dir)
        api_ref = api_ref.rename_columns({"api": "title", "ref": "contents"})
        concept_docs = lcstep.load_concept_docs(args.data_dir)
        concept_docs = concept_docs.rename_columns({"path": "title", "ref": "contents"})
        docs = concatenate_datasets([api_ref, concept_docs])
        logger.debug(f"collected {len(docs)} docs for vector store")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
        docs_store = rag.setup_store(
            logger,
            store,
            [
                {"title": d["title"][0], "chunk": i, "contents": t}
                for d in docs.iter(1)
                for i, t in enumerate(text_splitter.split_text(d["contents"][0]))
            ],
        )

        store.batch.wait_for_vector_indexing()
        system = rag.RAG(model, docs_store, args.k)
    elif system == "aag":
        store = weaviate.connect_to_local()

        # set up skill library from concept docs
        concept_docs = lcstep.load_concept_docs(args.data_dir)
        skills = asyncio.run(aag.build_concept_skills(model, concept_docs))
        aag.setup_skills(store, skills)

        # set up API ref store
        api_ref = lcstep.load_api_ref(args.data_dir)
        api_ref = api_ref.rename_column("ref", "documentation")
        aag.setup_api_ref(store, list(api_ref.iter(1)))

        store.batch.wait_for_vector_indexing()
        system = aag.AAG(model, store)
    else:
        raise NotImplementedError(args.system)

    print("running evaluation...", file=sys.stderr)
    asyncio.run(evaluate(system, data, min(args.workers, n)))
