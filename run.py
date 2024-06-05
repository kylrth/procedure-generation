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
import tempfile
import traceback
from pathlib import Path

import weaviate
from weaviate.embedded import EmbeddedOptions

import dataset
import retrieval
from dataset import Procedure
from systems import Model, System, aag, rag
from utils import log, spread_gather


async def generate_and_record(
    logger: logging.Logger,
    csv: log.CSVLogger,
    human: log.HumanLogger,
    model: System,
    id_: int,
    p: Procedure,
):
    """Generate a procedure for this item with the model, and log the result."""
    with human.for_id(id_) as hlog:
        try:
            hlog.write(f"processing query '{p.output}'\n")
            hlog.write(f"  input: '{p.input_}')'\n")
            res = await model.generate(hlog, p.output, p.input_)
            csv.result(
                log.Result(
                    ID=id_,
                    model=res.model,
                    gold=p,
                    completion=res.answer,
                )
            )
            hlog.write(f"BEGIN COMPLETION:\n{dataset.format_steps(res.answer)}\nEND COMPLETION\n")
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


class NiceWeaviate(weaviate.WeaviateClient):
    """This Weaviate client keeps things in a temporary directory and deletes it when done.

    It also sets nice defaults for using an embedded instance like this.
    """

    tdir: tempfile.TemporaryDirectory
    port: int
    grpc_port: int

    def __init__(self, port: int, grpc_port: int):
        self.tdir = tempfile.TemporaryDirectory()
        self.port = port
        self.grpc_port = grpc_port

    def __enter__(self):
        path = self.tdir.__enter__()
        super().__init__(
            embedded_options=EmbeddedOptions(
                persistence_data_path=path,
                version="1.25.1",
                port=self.port,
                additional_env_vars={
                    "AUTOSCHEMA_ENABLED": "false",
                    "DISABLE_TELEMETRY": "true",
                    "LOG_LEVEL": "warning",
                },
                grpc_port=self.grpc_port,
            )
        )
        super().__enter__()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self.tdir.__exit__(exc_type, exc_value, traceback)


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
    outdir = Path("output") / system_name / dataset_name / args.embedder

    human = log.HumanLogger(outdir)

    port = random.randint(1000, 65534)
    with NiceWeaviate(port, port + 1) as client:
        if system_name == "rag":
            # set up vector store with supporting docs + the train set of procedures
            logger.debug("RAG: collecting docs")
            docs = ds.docs(include_procedures=dataset.Split.TRAIN)
            logger.debug(f"RAG: collected {len(docs)} docs for vector store")

            logger.info("RAG: Creating collection and uploading docs to Weaviate collection")
            store = retrieval.DocStore(
                client, "Docs", "Documents for retrieval", args.embedder, cache_path
            )
            store.populate(logger, docs)

            system = rag.RAG(model, store, args.k, args.dataset)
        elif system_name == "aag":
            # set up vector store for unchunked train procedures
            logger.debug("AAG: collecting train procedures")
            procedures = ds.procedures(dataset.Split.TRAIN)
            logger.debug(f"AAG: collected {len(procedures)} procedures for skill library")

            logger.info("AAG: Creating collection and uploading procedures to Weaviate collection")
            store = retrieval.ProcedureStore(
                client,
                "Procedures",
                "Skill library",
                args.embedder,
                cache_path,
                retrieval.procedure_formatter_for(dataset_name),
            )
            store.populate(logger, procedures)

            system = aag.AAG(model, store, args.k, args.dataset)
        else:
            raise NotImplementedError(args.system)

        # shorten eval set according to -n
        val_data = ds.procedures(dataset.Split.VAL)
        n = min(args.n, len(val_data))
        val_data = val_data[:n]
        logger.info(f"loaded {len(val_data)} eval examples")

        logger.info("starting generation...")

        with log.CSVLogger(outdir / "output.csv") as csv:
            try:
                asyncio.run(
                    spread_gather(
                        lambda item: generate_and_record(logger, csv, human, system, *item),
                        enumerate(val_data),
                        min(args.workers, n),
                        len(val_data),
                    )
                )
                logger.info(f"see results in in ./{outdir}")
            finally:
                store.embedder.flush()
