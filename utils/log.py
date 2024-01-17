import csv
import json
import os
import sys
import textwrap
import typing
from dataclasses import dataclass
from pathlib import Path

from langchain.base_language import BaseLanguageModel
from langchain.schema import BaseMessage


@dataclass
class Result:
    """A structured object containing the information to log for a generated response to a query."""

    ID: int
    source: str
    query: str
    label: str
    prompt: str | list[BaseMessage]
    completions: list[str]
    retrieved_docs: list[dict[str, str | float]]
    context: str
    model: BaseLanguageModel


class ResultsLogger:
    """This object logs results to a CSV as well as a more human-readable log folder."""

    _field_names: typing.ClassVar[list[str]] = []

    def __init__(self, csv_path: str | os.PathLike | None, folder_path: str | os.PathLike | None):
        self.csv = CSVLogger(csv_path) if csv_path is not None else None
        self.human = HumanLogger(folder_path) if folder_path is not None else None

    def __enter__(self):
        if self.csv:
            self.csv.__enter__()

        return self

    def result(self, r: Result):
        """Log the response(s) generated for a particular item."""
        if self.csv:
            self.csv.result(r)
        if self.human:
            self.human.result(r)

    def exception(self, _id: int, msg: str):
        """Log an exception encountered while working on a particular item."""
        if self.human:
            self.human.exception(_id, msg)

    def evaluation(self, _id: int, scores: dict[str, typing.Any]):
        """Log evaluation results for a particular item."""
        if self.human:
            self.human.evaluation(_id, scores)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.csv:
            return self.csv.__exit__(exc_type, exc_val, exc_tb)

        return False


class CSVLogger:
    """This object logs results to a CSV."""

    _field_names: typing.ClassVar[list[str]] = [
        "",
        "question_id",
        "question",
        "labeled_answer",
        "prompt",
        "response",
        "retrieved_docs",
        "context",
    ]

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)

    def __enter__(self):
        if self.path.exists():
            raise FileExistsError(str(self.path))

        self.f = self.path.open("w", newline="")
        self.w = csv.DictWriter(self.f, self._field_names)

        # track row number
        self.i = 0

        return self

    def result(self, r: Result):
        retrieved = json.dumps(r.retrieved_docs)
        prompt = r.prompt
        if isinstance(prompt, list):
            prompt = self._format_messages(prompt)

        for completion in r.completions:
            self.w.writerow(
                {
                    "": self.i,
                    "question_id": r.ID,
                    "question": r.prompt,
                    "labeled_answer": r.label,
                    "prompt": r.prompt,
                    "response": completion,
                    "retrieved_docs": retrieved,
                    "context": r.context,
                }
            )

            self.i += 1

    @staticmethod
    def _format_messages(messages: list[BaseMessage]) -> str:
        return "\n\n".join(msg.content for msg in messages)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.close()

        return False


class HumanLogger:
    """This object logs results to a human-readable log folder, along with any details of exceptions
    encountered."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        try:
            self.path.mkdir(parents=True)
        except FileExistsError as e:
            raise FileExistsError(f"log folder '{self.path}' already exists") from e

    def _get_log_path(self, _id: int) -> Path:
        return self.path / f"{_id}.log"

    def result(self, r: Result):
        with self._get_log_path(r.ID).open("w") as f:
            prompt = r.prompt
            if isinstance(prompt, list):
                prompt = self._format_messages(prompt)

            f.write(f"processing query '{r.query}' from '{r.source}'\n")
            f.write(f"retrieved {len(r.retrieved_docs)} docs for query '{r.query}'\n")
            f.write("prompt:\n")
            f.write(textwrap.indent(prompt, "  "))

            logged_completion = (
                "BEGIN COMPLETION:\n"
                + "END COMPLETION\nBEGIN COMPLETION".join(r.completions)
                + "\nEND COMPLETION"
            )
            f.write(f"completion:\n{logged_completion}\n")

            def count(msg: str | list[BaseMessage]) -> int:
                if isinstance(msg, list):
                    return r.model.get_num_tokens_from_messages(msg)

                return r.model.get_num_tokens(msg)

            f.write(
                f"used {count(prompt) * len(r.completions)} input tokens and "
                f"{sum(count(s) for s in r.completions)} output tokens\n"
            )
            f.write(f"got {len(r.completions)} completions\n")

    @staticmethod
    def _format_messages(messages: list[BaseMessage]) -> str:
        def _format(msg: BaseMessage) -> str:
            if "\n" not in msg.content:
                return msg.__class__.__name__ + "(" + msg.content + ")"
            return msg.__class__.__name__ + "(\n  " + msg.content.replace("\n", "\n  ") + "\n)"

        return "\n".join(_format(msg) for msg in messages)

    def exception(self, _id: int, msg: str):
        with self._get_log_path(_id).open("a+") as f:
            f.write(f"{msg}: {sys.exc_info()}\n")

    def evaluation(self, _id: int, scores: dict[str, typing.Any]):
        with self._get_log_path(_id).open("a+") as f:
            f.write("EVALUATION:\n")
            for metric in scores:
                f.write(f"  {metric}: {scores[metric]}\n")
