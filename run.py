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
import weaviate
from datasets import Dataset
from langchain.embeddings import HuggingFaceEmbeddings

import lcstep
from evaluation.eval import evaluation
from systems import RAG, FewShot, Model, System, aag
from systems.aag import AAG, setup_api_ref, setup_skills
from utils import spread_gather


def make_logger(for_file: str) -> logging.Logger:
    """Create a new logger that writes to logs/{name}.log, where `name` is converted from `for_file`
    with directory separators replaced with dashes."""
    name = for_file.replace("/", "-")
    log_file = Path("logs") / f"{name}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.propagate = False
    logger.addHandler(logging.FileHandler(log_file, "w"))
    logger.setLevel(logging.DEBUG)
    return logger


async def generate_and_evaluate(model: System, item: dict[str, Any]):
    """Generate a procedure for this item with the model, and then evaluate it."""
    goal = item["goal"][0]

    logger = make_logger(str(item["path"][0]))

    res = await model.agenerate(goal, logger)
    logger.debug(f"got {len(res)} generations")

    scores = defaultdict(list)
    for completion in res:
        try:
            evals = await evaluation(completion, item, logger)
        except Exception:
            logger.exception("exception during evaluation")
            break
        for metric in evals:
            scores[metric].append(evals[metric])

    score_print = "\n".join(f"  {metric}: {scores[metric]}" for metric in scores)
    logger.debug("final scores:\n" + score_print)

    return scores, item["path"][0]


async def evaluate(model: System, data: Dataset, n_workers: int = 10):
    """Evaluate the system with the given text generation dataset."""
    results = await spread_gather(
        lambda prompt: generate_and_evaluate(model, prompt),
        data.iter(1),
        n_workers,
        len(data),
    )

    # collect results
    scores = defaultdict(list)
    broken = []
    for evals, path in results:
        # if prompt had no correct completions
        if len(evals) == 0:
            broken.append(path)
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
        default="FewShot",
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

    rag_options = parser.add_argument_group("RAG")
    rag_options.add_argument(
        "--rag-embed-model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="HuggingFace model to use for embeddings",
    )
    rag_options.add_argument(
        "--rag-embed-n",
        type=int,
        default=sys.maxsize,
        help=(
            "number of samples to use in the vector store. This is ignored when the vectors/ "
            "directory exists"
        ),
    )
    rag_options.add_argument(
        "--rag-embed-gpu", action="store_true", help="compute embeddings on the GPU"
    )

    args = parser.parse_args()

    cache_path = Path("cache")

    print("loading data...", file=sys.stderr)
    data = lcstep.load_formatted_docs(args.data_dir)

    # reserve a few examples for few-shot
    examples = data.select(np.arange(0, args.k))
    data = data.select(np.arange(5, len(data)))

    # shorten dataset according to -n
    n = min(args.n, len(data))
    data = data.select(np.arange(0, n))

    print("creating system...", file=sys.stderr)
    model = Model.from_full_name(args.model)
    system = args.system.lower()
    if system == "fewshot":
        # prepare examples
        shots = []
        for _, item in zip(range(args.k), examples.iter(1), strict=False):
            proc = lcstep.Procedure(item["goal"][0], item["steps"][0], "").to_text()
            goal, steps = proc.strip().split("\n\n")
            print(goal, steps)
            shots.append((goal, steps))

        system = FewShot(model, shots=shots)
    elif system == "rag":
        api_refs = lcstep.load_api_ref(args.data_dir)
        concept_docs = lcstep.load_concept_docs(args.data_dir)
        embedder = HuggingFaceEmbeddings(
            model_name=args.rag_embed_model,
            encode_kwargs=None if args.rag_embed_gpu else {"device": "cpu"},
        )
        system = RAG(
            model,
            args.rag_k,
            api_refs,  # TODO handle api_refs and concept_docs
            embedder,
            args.rag_embed_n,
            cache_path / "embeddings" / args.rag_embed_model,
        )
    elif system == "aag":
        store = weaviate.connect_to_local()

        # set up skill library from concept docs
        concept_docs = lcstep.load_concept_docs(args.data_dir)
        skills = asyncio.run(aag.build_concept_skills(model, concept_docs))
        setup_skills(store, skills)

        # set up API ref store
        api_ref = lcstep.load_api_ref(args.data_dir)
        setup_api_ref(store, api_ref.iter(1))

        store.batch.wait_for_vector_indexing()
        system = AAG(model, store)
    else:
        raise NotImplementedError(args.system)

    print("running evaluation...", file=sys.stderr)
    asyncio.run(evaluate(system, data, min(args.workers, n)))
