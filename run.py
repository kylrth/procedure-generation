# ruff: noqa: T201
# This script needs to print.
# ruff: noqa: I001, E402  # need to shut up before importing langchain

import shutup

shutup.please()

import argparse
import asyncio
import logging
import random
import sys
import traceback
from pathlib import Path

import dataset
import retrieval
from dataset import LinearProcedure
from systems import System, AAG, FewShot, ReAct, RAG
from utils import log, spread_gather
from utils.weaviate import NiceWeaviate
import time
from model import Model


async def generate_and_record(
    logger: logging.Logger,
    csv: log.CSVLogger,
    human: log.HumanLogger,
    model: System,
    id_: int,
    p: LinearProcedure,
):
    """Generate a procedure for this item with the model, and log the result."""
    with human.for_id(id_) as hlog:
        try:
            hlog.write(f"processing query '{p.output}'\n")
            hlog.write(f"  input: '{p.input_}'\n")
            res = await model.generate(hlog, p.output, p.input_)
            hlog.write("\nFINISHED GENERATING\n\n")
            csv.result(
                log.Result(
                    ID=id_,
                    model=res.model,
                    gold=p,
                    completion=res.answer,
                )
            )
            hlog.write(f"BEGIN GENERATED:\n{dataset.format_steps(res.answer)}\nEND GENERATED\n")
            hlog.write(f"BEGIN REFERENCE:\n{p.format_steps()}\nEND REFERENCE\n")
            if res.input_tokens != -1 or res.output_tokens != -1:
                hlog.write(f"used {res.input_tokens} input and {res.output_tokens} output tokens\n")
        except Exception:  # noqa: BLE001  # logging the exception for tracing purposes
            hlog.write(f"EXCEPTION for id {id_}: {traceback.format_exc()}\n")
            logger.error(  # noqa: TRY400
                f"exception for item {id_}; see ./{hlog.name} for details\n"
            )


def int_leq(v: int):
    """Custom type conversion function which validates that the int is less than or equal to v."""

    def validate(value):
        i = int(value)
        if i > v:
            raise argparse.ArgumentTypeError(f"{value} is not less than {v}")

        return i

    return validate


async def main(args):
    logger = logging.getLogger("main")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

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

    cache_path = Path("cache") / system_name / dataset_name / args.embedder

    out_name = (
        system_name
        + ("_no-summ" if system_name == "aag" and not args.summarize else "")
        + ("_no-critic" if system_name in ("rag", "aag") and not args.critic else "")
        + (f"_{args.n_queries}" if system_name == "aag" and args.n_queries != 4 else "")
    )
    outdir = Path("output") / out_name / dataset_name / args.embedder

    human = log.HumanLogger(outdir)
    store = None

    async with NiceWeaviate() as client:
        if system_name == "zeroshot":
            system = FewShot(model, args.dataset, shots=None)
        elif system_name == "fewshot":
            logger.debug(f"FewShot: selecting {args.k} docs")
            procedures = ds.graphs(dataset.Split.TRAIN)
            rng = random.Random(27)
            shots = [str(shot) for shot in rng.sample(procedures, args.k)]

            system = FewShot(model, args.dataset, shots)
        elif system_name == "rag":
            # set up vector store with supporting docs + the train set of procedures
            logger.debug("RAG: collecting docs")
            procedures = ds.graphs(dataset.Split.TRAIN | dataset.Split.VAL)
            logger.debug(f"RAG: collected {len(procedures)} docs for vector store")

            logger.info("RAG: Creating collection and uploading docs to Weaviate collection")
            embedder = retrieval.CachingEmbedder(
                retrieval.embedder_from_name(args.embedder), cache_path
            )
            store = retrieval.GraphProcedureStore(client, embedder)
            await store.populate(logger, procedures)

            system = RAG(model, store, args.k, args.dataset, args.critic, args.hs)
        elif system_name in {"react", "aag"}:
            fancy = "AAG" if system_name == "aag" else "ReAct"

            # set up vector store for unchunked train procedures
            logger.debug(f"{fancy}: collecting train procedures")
            procedures = ds.graphs(dataset.Split.TRAIN | dataset.Split.VAL)
            logger.debug(f"{fancy}: collected {len(procedures)} procedures for skill library")

            logger.info(
                f"{fancy}: Creating collection and uploading procedures to Weaviate collection"
            )
            embedder = retrieval.CachingEmbedder(
                retrieval.embedder_from_name(args.embedder), cache_path
            )
            store = retrieval.GraphProcedureStore(
                client,
                embedder,
            )
            await store.populate(logger, procedures)

            if system_name == "aag":
                system = AAG(
                    model,
                    store,
                    args.k,
                    args.dataset,
                    args.summarize,
                    args.critic,
                    args.hs,
                    args.n_queries,
                )
            else:  # "react"
                system = ReAct(model, args.dataset, store, args.k, args.hs)
        else:
            raise NotImplementedError(args.system)

        # shorten eval set according to -n
        eval_data = ds.graphs(dataset.Split.TEST)
        n = min(args.n, len(eval_data))
        eval_data = eval_data[:n]
        logger.info(f"loaded {len(eval_data)} eval examples")

        logger.info("starting generation...")

        with log.CSVLogger(outdir / "output.csv") as csv:
            try:
                start = time.time_ns()
                asyncio.run(
                    spread_gather(
                        lambda item: generate_and_record(logger, csv, human, system, *item),
                        enumerate(eval_data),
                        min(args.workers, n),
                        len(eval_data),
                    )
                )
                tot_time = time.time_ns() - start
                logger.info(f"see results in in ./{outdir}")
                logger.info(
                    f"runtime for {args.system} with critic {args.critic} on {args.dataset}: "
                    f"{tot_time / 1e9:.1f}s"
                )
            finally:
                if store is not None:
                    store.embedder.flush()


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
        default="openai-gpt-3.5-turbo-0125",
        help="full name of service & model to use",
    )
    parser.add_argument(
        "-em",
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
        help="max # examples to provide for FewShot, or max to retrieve for RAG/AAG",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=4,
        help="Number of re-written queries",
    )
    parser.add_argument(
        "--summarize", action="store_true", help="Whether to use summarization in the AAG"
    )
    parser.add_argument(
        "--critic", action="store_true", help="Whether to use critic in the system or not"
    )
    parser.add_argument(
        "--hs", action="store_true", help="Whether to use hierarchical search in the system or not"
    )
    args = parser.parse_args()

    asyncio.run(main(args))
