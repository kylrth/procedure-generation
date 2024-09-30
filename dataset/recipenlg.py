import csv
import itertools
import json
import math
import pickle
import random
import re
import sys
from os import PathLike
from typing import Any, Iterable, Sequence, cast

from .base import Dataset, Doc, GraphProcedure, LinearProcedure, Split


_step_prefixes = re.compile(r"^\s*(?:\d+(?:\.|\))\s*|-)\s*(.*)$")


def parse_steps(s: str) -> list[str]:
    # should be list of steps
    steps = json.loads(s)

    out = []
    for step in steps:
        if step == "":
            continue

        match = _step_prefixes.match(step)
        if match:
            out.append(match.group(1))
        else:
            out.append(step.strip())

    return out


def recipe_to_procedure(d: dict[str, Any]) -> LinearProcedure:
    return LinearProcedure(
        input_=", ".join(json.loads(d["ingredients"])),
        output=d["title"],
        steps=parse_steps(d["directions"]),
    )


class RecipeNLG(Dataset):
    """A random subset of size n of the RecipeNLG dataset."""

    def __init__(self, data_dir: str | PathLike, n: int = sys.maxsize):
        super().__init__(data_dir)
        self.reservoir = Reservoir(n, seed=42)
        self.g_train = None

    def _init_procedures(self) -> list[LinearProcedure]:
        with (self.dir / "RecipeNLG" / "full_dataset.csv").open(newline="") as f:
            header = f.readline()
            self.reservoir.sample(f)

        reader = csv.DictReader(itertools.chain((header,), self.reservoir.samples))
        out = []
        for row in reader:
            out.append(recipe_to_procedure(row))

        self.reservoir.rng.shuffle(out)

        return out

    def _init_graphs(self) -> list[GraphProcedure]:
        d = self.dir / "graphs" / "recipenlg"
        file_list = d.glob("*.pkl")
        graph_list: list[GraphProcedure] = []
        for file in file_list:
            with file.open("rb") as f:
                graph = pickle.load(f)
            graph_list.append(graph)

        # move all short recipes (<3 nodes) from val/test into train because they are often
        # low-quality
        # This means we need to keep track of these ourselves instead of letting Dataset do it,
        # because we're changing the ratios.
        self.g_train = Split.TRAIN.get(graph_list)
        self.g_val: list[GraphProcedure] = []
        self.g_test: list[GraphProcedure] = []

        for ex in Split.VAL.get(graph_list):
            if ex.counts()[0] >= 3:
                self.g_val.append(ex)
        for ex in Split.TEST.get(graph_list):
            if ex.counts()[0] >= 3:
                self.g_test.append(ex)

        return []  # not used, we override the graphs method

    def graphs(self, split: Split) -> Sequence[GraphProcedure]:
        if self.g_train is None:
            self._init_graphs()

        # we override so we can maintain the separate lists; see _init_graphs
        out = []
        for s in split:
            match s:
                case Split.TRAIN:
                    out.extend(cast(Sequence[GraphProcedure], self.g_train))
                case Split.VAL:
                    out.extend(self.g_val)
                case Split.TEST:
                    out.extend(self.g_test)
        return out

    def _get_docs(self) -> list[Doc]:
        # TODO get generic cooking material
        return []


class Reservoir:
    """Implements "algorithm L" for reservoir sampling as described here:
    https://en.wikipedia.org/wiki/Reservoir_sampling#Optimal:_Algorithm_L

    Closely follows what was written here:
    https://github.com/alexprengere/reservoir/blob/master/reservoir.py
    but with the nice interface from here:
    https://github.com/mattiaciollaro/reservoir/blob/master/reservoir.py
    """

    size: int
    seen: int
    samples: list
    rng: random.Random

    def __init__(self, size: int, seed=None):
        self.size = size
        self.seen = 0
        self.samples = []
        self.rng = random.Random(seed)

    def sample(self, i: Iterable):
        """Performs reservoir sampling from i. Samples are placed in self.samples.

        If this method is called multiple times, self.samples will be a uniformly random subset of
        items from all these iterables.
        """
        gap_threshold = 4 * self.size

        iterator = iter(i)
        try:
            while True:
                self.seen += 1
                item = next(iterator)
                if len(self.samples) < self.size:
                    self.samples.append(item)
                elif self.seen < gap_threshold:
                    k = int(self.rng.random() * self.seen)
                    if k < self.size:
                        self.samples[k] = item
                else:
                    gap = int(math.log(self.rng.random()) / math.log(1 - self.size / self.seen))
                    self.seen += gap
                    for _ in range(gap):
                        item = next(iterator)
                    k = int(self.rng.random() * self.size)
                    self.samples[k] = item
        except StopIteration:
            pass
