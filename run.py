# ruff: noqa: T201
# This script needs to print.
# ruff: noqa: I001, E402  # need to shut up before importing langchain

import shutup

shutup.please()

import asyncio
import logging
import random
import sys
import time
import traceback
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import Sequence

from weaviate import WeaviateAsyncClient

import dataset
import retrieval
from dataset import GraphProcedure
from model import Model
from retrieval.embedder import Embedder
from systems import AAG, RAG, FewShot, ReAct, System
from utils import log, spread_gather
from utils.experiment import Config, help
from utils.weaviate import NiceWeaviate


async def generate_and_record(
    logger: logging.Logger,
    human: log.HumanLogger,
    model: System,
    id_: int,
    p: dataset.GraphProcedure,
):
    """Generate a procedure for this item with the model, and log the result."""
    with human.for_id(id_) as hlog:
        try:
            hlog.write(f"processing query '{p.get_title()}'\n")
            hlog.write(f"  input: '{p.get_inputs()}'\n")
            res = await model.generate(hlog, p.get_title(), p.get_inputs())
            hlog.write("\nFINISHED GENERATING\n\n")

            hlog.write(f"BEGIN GENERATED:\n{res.answer!s}\nEND GENERATED\n")
            hlog.write(f"BEGIN REFERENCE:\n{p!s}\nEND REFERENCE\n")
            if res.input_tokens != -1 or res.output_tokens != -1:
                hlog.write(f"used {res.input_tokens} input and {res.output_tokens} output tokens\n")

            # pickle the reference and generated graph objects for later evaluation
            hlog.result(p, res.answer)
        except Exception:  # noqa: BLE001  # logging the exception for tracing purposes
            hlog.write(f"EXCEPTION for id {id_}: {traceback.format_exc()}\n")
            logger.error(  # noqa: TRY400
                f"exception for item {id_}; see ./{hlog.name} for details\n"
            )


class DatasetOption(StrEnum):
    LCStep = auto()
    RecipeNLG = auto()
    CHAMP = auto()

    def make(self, data_dir: Path) -> dataset.Dataset:
        match self:
            case DatasetOption.LCStep:
                return dataset.LCStep(data_dir)
            case DatasetOption.RecipeNLG:
                return dataset.RecipeNLG(data_dir, n=10000)
            case DatasetOption.CHAMP:
                return dataset.CHAMP(data_dir)


class SystemOption(StrEnum):
    ZeroShot = auto()
    FewShot = auto()
    RAG = auto()
    ReAct = auto()
    AAG = auto()

    async def make(
        self,
        logger: logging.Logger,
        exp: "Experiment",
        model: Model,
        train: Sequence[GraphProcedure],
        embedder: Embedder,
        client: WeaviateAsyncClient,
    ) -> System:
        """Create a System ready to answer queries."""
        match self:
            case SystemOption.ZeroShot:
                return FewShot(model, exp.ds, shots=None)
            case SystemOption.FewShot:
                logger.debug(f"FewShot: selecting {exp.k} ICL examples from training data")
                rng = random.Random(27)
                shots = [str(shot) for shot in rng.sample(train, exp.k)]

                return FewShot(model, exp.ds, shots)

        logger.info(f"{self.fancy_name}: creating Weaviate collection")
        store = retrieval.GraphProcedureStore(client, embedder)
        await store.setup_collection()

        logger.info(f"{self.fancy_name}: uploading {len(train)} procedures to Weaviate collection")
        await store.populate(logger, train)

        match self:
            case SystemOption.RAG:
                return RAG(model, store, exp.k, exp.ds, exp.critic, exp.hierarchical_search)
            case SystemOption.ReAct:
                return ReAct(model, exp.ds, store, exp.k, exp.hierarchical_search)
            case SystemOption.AAG:
                return AAG(
                    model, store, exp.k, exp.ds, exp.critic, exp.hierarchical_search, exp.n_queries
                )

    @property
    def fancy_name(self) -> str:
        match self:
            case SystemOption.ZeroShot:
                return "ZeroShot"
            case SystemOption.FewShot:
                return "FewShot"
            case SystemOption.RAG:
                return "RAG"
            case SystemOption.ReAct:
                return "ReAct"
            case SystemOption.AAG:
                return "AAG"

    @property
    def uses_retrieval(self) -> bool:
        match self:
            case SystemOption.ZeroShot | SystemOption.FewShot:
                return False
            case _:
                return True


# ruff: noqa: RUF009  # here we use a helper function to set field info
@dataclass
class Experiment(Config):
    ds: DatasetOption = help(choices=list(DatasetOption), help="dataset to run the system on")
    system: SystemOption = help(choices=list(SystemOption), help="system to perform generation")
    model: str = help(
        default="openai-gpt-3.5-turbo-0125", help="full name of service/model for completions"
    )
    embedder: str = help(
        default="hf-all-mpnet-base-v2", help="full name of service/model for embeddings"
    )
    n: int = help(default=sys.maxsize, help="limit the number of samples to test")
    workers: int = help(default=10, help="number of concurrent queries to process")

    # method-specific options
    k: int = help(
        default=3, help="max # examples for FewShot, or max to retrieve for retrieval-based methods"
    )
    n_queries: int = help(default=4, help="number of rewritten queries for AAG")
    critic: bool = help(default=None, help="whether to use the critic")
    hierarchical_search: bool = help(default=None, help="whether to use hierarchical search")

    dataset_root: Path = help(default=Path("dataset"), help="path to dataset dir")
    emb_cache_root: Path = help(default=Path("cache"), help="path to cache dir")
    logdir_root: Path = help(default=Path("output"), help="path to output dir")

    def __post_init__(self):
        if not self.system.uses_retrieval:
            if self.critic or self.hierarchical_search:
                problem = ""
                if self.critic:
                    problem = "critic"
                elif self.hierarchical_search:
                    problem = "hs"
                raise TypeError(f"cannot use --{problem} with --system {self.system}")

        else:  # RAG, ReAct, AAG
            if self.critic is None:
                self.critic = self.system is SystemOption.AAG
            if self.hierarchical_search is None:
                self.hierarchical_search = self.system is SystemOption.AAG

    @staticmethod
    def make_logger() -> logging.Logger:
        logger = logging.getLogger("main")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        return logger

    def make_embedder(self) -> retrieval.CachingEmbedder:
        cache_path = self.emb_cache_root / self.system / self.ds / self.embedder
        return retrieval.CachingEmbedder(retrieval.embedder_from_name(self.embedder), cache_path)

    @property
    def logdir(self) -> Path:
        # descriptive output directory name
        out_name = str(self.system)
        match self.system:
            case SystemOption.RAG | SystemOption.ReAct:
                if self.hierarchical_search:
                    out_name += "_hs"
                if self.critic:
                    out_name += "_critic"
            case SystemOption.AAG:
                if not self.hierarchical_search:
                    out_name += "_nohs"
                if not self.critic:
                    out_name += "_nocritic"
                if self.n_queries != self.__class__.n_queries:
                    out_name += "_q" + str(self.n_queries)

        return self.logdir_root / out_name / self.ds / self.embedder

    async def run(self):
        logger = self.make_logger()
        human = log.HumanLogger(self.logdir)

        logger.info("loading dataset...")
        ds = self.ds.make(self.dataset_root)

        # shorten eval set according to -n
        eval_data = ds.graphs(dataset.Split.VAL)
        n = min(self.n, len(eval_data))
        eval_data = eval_data[:n]
        logger.info(f"loaded {len(eval_data)} eval examples")

        logger.info("creating system...")
        model = Model.from_full_name(self.model)
        with self.make_embedder() as emb:
            async with NiceWeaviate() as client:
                system = await self.system.make(
                    logger, self, model, ds.graphs(dataset.Split.TRAIN), emb, client
                )

                logger.info("starting generation...")

                start = time.time_ns()
                await spread_gather(
                    lambda item: generate_and_record(logger, human, system, *item),
                    enumerate(eval_data),
                    min(self.workers, n),
                    len(eval_data),
                )
                tot_time = time.time_ns() - start

                logger.info(f"runtime for {self.system} on {self.ds}: {tot_time / 1e9:.1f}s")
                logger.info(f"see results in in ./{self.logdir}")


if __name__ == "__main__":
    exp = Experiment.from_args()
    print(exp, file=sys.stderr)
    asyncio.run(exp.run())
