"""Present the pairwise evaluation results between two methods next to aggregates of the heuristic
evaluations."""

import csv
import itertools
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum, StrEnum, auto
from pathlib import Path

from run import DatasetOption
from utils.experiment import Config, chelp


class OldSystemOption(StrEnum):
    # TODO convert this to the new system once we have results for the new system
    ZeroShot = auto()
    FewShot = auto()
    RAG = auto()  # has critic
    RAGNoCritic = "rag_no-critic"
    AAG = auto()
    AAGNoCritic = "aag_no-critic"
    AAGNoSumm = "aag_no-summ"
    AAGNoSummNoCritic = "aag_no-summ_no-critic"


class PairwiseResult(IntEnum):
    """See get_final_choice in run.py."""

    Tie = 0
    First = 1
    Second = 2
    NoResult = -1

    def reverse(self) -> "PairwiseResult":
        match self:
            case PairwiseResult.First:
                return PairwiseResult.Second
            case PairwiseResult.Second:
                return PairwiseResult.First
        return self


@dataclass
class Comparison(Config):
    ds: DatasetOption = chelp(choices=list(DatasetOption), help="dataset to show results for")
    s1: OldSystemOption = chelp(choices=list(OldSystemOption), help="first system to compare")
    s2: OldSystemOption = chelp(choices=list(OldSystemOption), help="second system to compare")
    emb: str = chelp(default="hf-all-mpnet-base-v2", help="embedder used for both systems")

    dataset_root: Path = chelp(default=Path("dataset"), help="path to dataset dir")
    logdir_root: Path = chelp(default=Path("output"), help="path to output dir")

    @staticmethod
    def get_eval_results(f: Path) -> dict[int, dict[str, float]]:
        """Read the CSV and flatten the ROUGE scores out so all results are in a dict for each
        question ID."""
        d = {}
        with f.open(newline="") as csvf:
            r = csv.DictReader(csvf)
            for row in r:
                id_ = int(row["_id"])
                del row["_id"]

                # flatten ROUGE dict
                if "ROUGE" in row:
                    rouge_dict = json.loads(row["ROUGE"].replace("'", '"'))
                    for k in rouge_dict:
                        row[k] = rouge_dict[k]
                    del row["ROUGE"]

                d[id_] = {k: float(row[k]) for k in row}

        return d

    @staticmethod
    def get_pairwise_results(f: Path, *, reverse: bool = False) -> dict[int, PairwiseResult]:
        """Read the CSV and return the preference value for each."""
        d = {}
        with f.open(newline="") as csvf:
            r = csv.DictReader(csvf)
            for row in r:
                id_ = int(row["question_id"])
                choice = PairwiseResult(int(row["choice"]))

                if reverse:
                    choice = choice.reverse()

                d[id_] = choice

        return d

    @staticmethod
    def make_logger() -> logging.Logger:
        logger = logging.getLogger("main")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        return logger

    def collect(
        self,
    ) -> tuple[dict[str, list[float]], dict[str, list[float]], list[PairwiseResult]]:
        """Collect the results for each evaluation for both systems, as well as their pairwise
        evaluations."""
        f_eval1 = self.logdir_root / self.s1 / self.ds / self.emb / "eval_results.csv"
        f_eval2 = self.logdir_root / self.s2 / self.ds / self.emb / "eval_results.csv"
        eval1 = self.get_eval_results(f_eval1)
        eval2 = self.get_eval_results(f_eval2)

        f_pairwise = self.logdir_root / f"{self.s1}_{self.s2}_{self.ds}_{self.emb}_pair_eval.csv"
        reverse = False
        if not f_pairwise.exists():
            f_pairwise = (
                self.logdir_root / f"{self.s2}_{self.s1}_{self.ds}_{self.emb}_pair_eval.csv"
            )
            reverse = True
        pairwise = self.get_pairwise_results(f_pairwise, reverse=reverse)

        logger = self.make_logger()

        # collect results into lists for each evaluation, skipping IDs with any missing evals
        res1: dict[str, list[float]] = defaultdict(list)  # map evaluation name to list of results
        res2: dict[str, list[float]] = defaultdict(list)  # map evaluation name to list of results
        respw: list[PairwiseResult] = []  # list of pairwise results
        for i in range(max(itertools.chain(eval1.keys(), eval2.keys(), pairwise.keys()))):
            try:
                one = eval1[i]
                two = eval2[i]
                pw = pairwise[i]
            except KeyError:
                # skip IDs missing any evaluations
                logger.debug(f"skipping ID {i} due to missing evaluation result")
                continue

            for k in one:
                res1[k].append(one[k])
            for k in two:
                res2[k].append(two[k])
            respw.append(pw)

        if sorted(res1.keys()) != sorted(res2.keys()):
            raise ValueError(
                f"evaluations for system 1 and 2 were different: {res1.keys()} vs. {res2.keys()}"
            )

        return res1, res2, respw


if __name__ == "__main__":
    c = Comparison.from_args()
    c.collect()
