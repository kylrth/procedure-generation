import csv
import io
import json
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Iterable

from langchain_core.messages import BaseMessage

from dataset import Procedure


@dataclass
class Result:
    """A structured object containing the results for a particular system generation."""

    ID: int
    model: str
    gold: Procedure
    completion: list[str]


class CSVLogger:
    """This object logs results to a CSV, for later evaluation."""

    _field_names: ClassVar[list[str]] = [
        "question_id",
        "input",
        "output",
        "gold_steps",
        "completion",
    ]

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)

    def __enter__(self):
        if self.path.exists():
            raise FileExistsError(str(self.path))

        self.f = self.path.open("w", newline="")
        self.w = csv.DictWriter(self.f, self._field_names)
        self.w.writeheader()

        return self

    def result(self, r: Result):
        self.w.writerow(
            {
                "question_id": r.ID,
                "input": r.gold.input_,
                "output": r.gold.output,
                "gold_steps": json.dumps(r.gold.steps),
                "completion": json.dumps(r.completion),
            }
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.close()

        return False


# for logging while generating results for a particular instance
class InstanceLogger:
    _wrapped: io.TextIOWrapper

    def __init__(self, wrapped: io.TextIOWrapper):
        self._wrapped = wrapped

    @property
    def name(self) -> str:
        return self._wrapped.name

    def __enter__(self):
        self._wrapped = self._wrapped.__enter__()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._wrapped.__exit__(exc_type, exc_value, traceback)

    def write(self, s: str, /) -> int:
        return self._wrapped.write(s)

    def writelines(self, lines: Iterable[str], /):
        return self._wrapped.writelines(lines)

    def log_prompt(self, p: str | list[BaseMessage], kind: str = "PROMPT"):
        self._wrapped.write(f"BEGIN {kind}\n")
        self._wrapped.write(textwrap.indent(messages_to_string(p), "  "))
        self._wrapped.write(f"\nEND {kind}\n")


class HumanLogger:
    """This object logs results to a human-readable log folder, with separate files for each
    instance."""

    def __init__(self, parent_path: str | os.PathLike):
        self.path = Path(parent_path)
        try:
            self.path.mkdir(parents=True)
        except FileExistsError as e:
            raise FileExistsError(f"log folder '{self.path}' already exists") from e

    def for_id(self, id_: int) -> InstanceLogger:
        """Return a logger for the specified example ID."""
        return InstanceLogger((self.path / f"{id_}.log").open("w"))


def messages_to_string(messages: str | list[BaseMessage]) -> str:
    """Convert what might be a list of chat messages into a loggable string."""
    if isinstance(messages, str):
        return messages

    def _format(msg: BaseMessage) -> str:
        if "\n" not in msg.content:
            return msg.__class__.__name__ + "(" + msg.content + ")"
        return msg.__class__.__name__ + "(\n  " + msg.content.replace("\n", "\n  ") + "\n)"

    return "\n".join(_format(msg) for msg in messages)
