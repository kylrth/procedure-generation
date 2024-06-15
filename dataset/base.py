from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Flag, auto
from os import PathLike
from pathlib import Path
from typing import Any


@dataclass
class Doc:
    """The base definition of a supporting document, as used across all datasets."""

    title: str
    contents: str

    def json(self):
        return {
            "title": self.title,
            "contents": self.contents,
        }


def format_steps(steps: list[str]) -> str:
    """Format steps of a procedure."""
    out = ""

    for i, step in enumerate(steps):
        out += f"{i+1}. {step}\n"

    return out[:-1]  # remove final newline


@dataclass
class Procedure:
    """The base definition of a procedure as used across all datasets."""

    input_: str
    output: str
    steps: list[str]

    def format_steps(self) -> str:
        """Format just the steps of the procedure."""
        return format_steps(self.steps)

    def to_doc(self) -> Doc:
        """Represent the procedure as a Doc."""
        return Doc(
            title=self.output + " using " + self.input_,
            contents=self.format_steps(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the procedure as a dictionary."""
        return {
            "input": self.input_,
            "output": self.output,
            "steps": self.steps,
        }


def train_val_test(data: list, val: float, test: float) -> tuple[list, list, list]:
    """Take test% of samples as test data from the end, then val% for val data, then everything
    before as train data.

    If you need a random grouping, shuffle beforehand.
    """
    if val < 0.0 or test < 0.0 or val + test > 1.0:
        raise ValueError(f"invalid val percentage {val} or test percentage {test}")

    train_len = int(len(data) * (1.0 - val - test))
    val_len = int(len(data) * val)

    train_set = data[:train_len]
    val_set = data[train_len : train_len + val_len]
    test_set = data[train_len + val_len :]

    return train_set, val_set, test_set


class Split(Flag):
    """Enum classes for specifying and selecting subsets of the data."""

    TRAIN = auto()
    VAL = auto()
    TEST = auto()

    def get(self, data: list, val: float = 0.1, test: float = 0.2) -> list:
        """Get the specified split of the given data."""
        d_train, d_val, d_test = train_val_test(data, val, test)

        out = []
        for subset in self:
            match subset:
                case Split.TRAIN:
                    out.extend(d_train)
                case Split.VAL:
                    out.extend(d_val)
                case Split.TEST:
                    out.extend(d_test)
                case _:
                    raise ValueError(f"unknown data split {self}")

        return out


class Dataset(ABC):
    """This is the interface that a dataset implements in order to be evaluated on."""

    def __init__(self, data_dir: str | PathLike):
        """Initialize the dataset object given the path to the data directory.

        This is usually just "./dataset".
        """
        self.dir = Path(data_dir)
        self._all = None

    @abstractmethod
    def _init_procedures(self) -> list[Procedure]:
        """Return the full list of procedures. Will only be called once and the result will be
        cached."""

    def procedures(self, split: Split) -> list[Procedure]:
        """Return the list of procedures corresponding to the specified section of data.

        May also be overridden if different functionality is desired.
        """
        if self._all is None:
            self._all = self._init_procedures()

        return split.get(self._all)

    @abstractmethod
    def _get_docs(self) -> list[Doc]:
        """Return all supporting documents."""

    def docs(self, include_procedures: Split | None = None) -> list[Doc]:
        """Return any supporting documents, optionally including a set of procedures as well.

        The procedures must be somehow converted to Doc objects.
        """
        out = self._get_docs()

        if include_procedures is None:
            return out

        procedures = self.procedures(include_procedures)
        for p in procedures:
            out.append(p.to_doc())

        return out
