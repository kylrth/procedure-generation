import csv
import itertools
import json
import math
import random
import re
import sys
from os import PathLike
from typing import Any, Iterable

from .base import Dataset, Doc, Procedure


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


def recipe_to_procedure(d: dict[str, Any]) -> Procedure:
    return Procedure(
        input_=", ".join(json.loads(d["ingredients"])),
        output=d["title"],
        steps=parse_steps(d["directions"]),
    )


class RecipeNLG(Dataset):
    """A random subset of size n of the RecipeNLG dataset."""

    def __init__(self, data_dir: str | PathLike, n: int = sys.maxsize):
        super().__init__(data_dir)
        self.reservoir = Reservoir(n, seed=42)

    def _init_procedures(self) -> list[Procedure]:
        with (self.dir / "RecipeNLG" / "full_dataset.csv").open(newline="") as f:
            header = f.readline()
            self.reservoir.sample(f)

        reader = csv.DictReader(itertools.chain((header,), self.reservoir.samples))
        out = []
        for row in reader:
            out.append(recipe_to_procedure(row))

        self.reservoir.rng.shuffle(out)

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
