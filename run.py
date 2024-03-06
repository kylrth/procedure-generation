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
from weaviate.embedded import EmbeddedOptions
from datasets import Dataset, concatenate_datasets
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dataset import lcstep
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

def modify_procedure_object_structure_for_RAG(data):
    def combine_input_and_steps(example):
        example["contents"] = example["input"] + "\n \n" + ' '.join(example["steps"])
        return example
    
    new_column = ["TEMP"] * len(data)
    data = data.add_column("contents", new_column)
    updated_data = data.map(combine_input_and_steps)
    updated_data = updated_data.rename_column("output", "title")
    updated_data = updated_data.select_columns(["title", "contents"])
    return updated_data

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
        "--dataset", 
        type=str, 
        default="", 
        choices=["LCSTEP", "RECIPE_NLG"], help="Dataset to run the system on")

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
    if args.dataset == "LCSTEP":
        data = lcstep.load_formatted_docs(args.data_dir)
        data_splits = data.train_test_split(test_size=0.3, shuffle=False)
        train_data = data_splits["train"]

    # shorten dataset according to -n
    n = min(args.n, len(train_data))
    train_data = train_data.select(np.arange(0, n))

    logger.info("creating system...")
    model = Model.from_full_name(args.model)
    system = args.system.lower()
    if system == "rag":
        client = weaviate.WeaviateClient(
                    embedded_options=EmbeddedOptions(
                        additional_env_vars={
                            "ENABLE_MODULES": "backup-filesystem,text2vec-openai,text2vec-cohere,text2vec-huggingface,ref2vec-centroid,generative-openai,qna-openai",
                            "BACKUP_FILESYSTEM_PATH": "/tmp/backups"
                        }
                    )
                )

        client.connect()
        # set up vector store with API refs and concept docs
        api_ref = lcstep.load_api_ref(args.data_dir)
        api_ref = api_ref.rename_columns({"api": "title", "ref": "contents"})
        concept_docs = lcstep.load_concept_docs(args.data_dir)
        concept_docs = concept_docs.rename_columns({"path": "title", "ref": "contents"})
        procedure_objects_transformed = modify_procedure_object_structure_for_RAG(train_data)
        docs = concatenate_datasets([procedure_objects_transformed, api_ref, concept_docs])
        logger.debug(f"collected {len(docs)} docs for vector store")
        
        docs_store = rag.setup_store(
            logger,
            client,
            store_name="Docs",
            store_desc="Documentation for the LangChain Python library."
            )

        if len(docs_store) == 0:
            logger.info(f"uploading {len(docs)} chunks to Weaviate collection")
            utils.populate_vector_store(logger, client, docs, collection_name="Docs")

        client.batch.wait_for_vector_indexing()
        system = rag.RAG(model, docs_store, args.k)
    elif system == "aag":
        store = weaviate.connect_to_local()

        # set up API ref store
        api_ref = lcstep.load_api_ref(args.data_dir)
        api_ref = api_ref.rename_column("ref", "documentation")
        aag.setup_api_ref(logger, store, list(api_ref.iter(1)))

        # set up skill library from concept docs
        concept_docs = lcstep.load_concept_docs(args.data_dir)
        skills = asyncio.run(aag.build_concept_skills(model, concept_docs))
        aag.setup_skills(logger, store, skills)

        store.batch.wait_for_vector_indexing()
        system = aag.AAG(model, store)
    else:
        raise NotImplementedError(args.system)

    print("running evaluation...", file=sys.stderr)
    asyncio.run(evaluate(system, data, min(args.workers, n)))
