import csv
import json
import random
from itertools import islice
from os import PathLike
from typing import Any

from .base import Dataset, Doc, Procedure


def recipe_to_procedure(d: dict[str, Any]) -> Procedure:
    return Procedure(
        _input=", ".join(json.loads(d["ingredients"])),
        output=d["title"],
        steps=json.loads(d["directions"]),
    )


class RecipeNLG(Dataset):
    def __init__(self, data_dir: str | PathLike, n: int | None = None):
        super().__init__(data_dir)
        self.n = n
        self.rng = random.Random(42)

    def _init_procedures(self) -> list[Procedure]:
        out = []
        with (self.dir / "RecipeNLG" / "full_dataset.csv").open(newline="") as f:
            reader = csv.DictReader(f)
            for row in islice(reader, self.n):
                out.append(recipe_to_procedure(row))

        self.rng.shuffle(out)

        return out

    def _get_docs(self) -> list[Doc]:
        # TODO get generic cooking material
        return []
