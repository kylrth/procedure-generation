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

from dataset import Doc, Procedure


@dataclass
class Result:
    """A structured object containing the information to log for a generated response to a query."""

    ID: int
    gold: Procedure
    prompt: str | list[BaseMessage]
    completion: list[str]
    retrieved_docs: list[Doc] | None
    context: str | None
    model: BaseLanguageModel


class ResultsLogger:
    """This object logs results to a CSV as well as a more human-readable log folder."""

    def __init__(self, path: str | os.PathLike):
        path = Path(path)
        csv_path = path / "output.csv"
        folder_path = path / "logs"

        self.csv = CSVLogger(csv_path)
        self.human = HumanLogger(folder_path)

    def __enter__(self):
        self.csv.__enter__()

        return self

    def result(self, r: Result):
        """Log the response(s) generated for a particular item."""
        self.csv.result(r)
        self.human.result(r)

    def exception(self, _id: int, msg: str):
        """Log an exception encountered while working on a particular item."""
        self.human.exception(_id, msg)

    def evaluation(self, _id: int, scores: dict[str, typing.Any]):
        """Log evaluation results for a particular item."""
        self.human.evaluation(_id, scores)

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.csv.__exit__(exc_type, exc_val, exc_tb)


class CSVLogger:
    """This object logs results to a CSV."""

    _field_names: typing.ClassVar[list[str]] = [
        "question_id",
        "input",
        "output",
        "gold_steps",
        "prompt",
        "completion",
        "retrieved_docs",
        "context",
        "model",
    ]

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)

    def __enter__(self):
        if self.path.exists():
            raise FileExistsError(str(self.path))

        self.f = self.path.open("w", newline="")
        self.w = csv.DictWriter(self.f, self._field_names)

        return self

    def result(self, r: Result):
        prompt = r.prompt
        if isinstance(prompt, list):
            prompt = "\n\n".join(msg.content for msg in prompt)

        docs = ""
        if r.retrieved_docs:
            docs = json.dumps([doc.json() for doc in r.retrieved_docs])

        self.w.writerow(
            {
                "question_id": r.ID,
                "input": r.gold._input,
                "output": r.gold.output,
                "gold_steps": json.dumps(r.gold.steps),
                "prompt": prompt,
                "completion": json.dumps(r.completion),
                "retrieved_docs": docs,
                "context": r.context,
                "model": r.model,
            }
        )

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

            f.write(f"processing query '{r.gold.output} ({r.gold._input})'\n")
            if r.retrieved_docs:
                f.write(f"retrieved {len(r.retrieved_docs)} docs\n")
            f.write(f"prompt to model {type(r.model).__name__}:\n")
            f.write(textwrap.indent(prompt, "  ") + "\n")

            generated = Procedure(r.gold._input, r.gold.output, r.completion)
            f.write(f"BEGIN COMPLETION:\n{generated.format_steps()}\nEND COMPLETION\n")
            f.write(f"BEGIN REFERENCE:\n{r.gold.format_steps()}\nEND REFERENCE\n")

            def count(msg: str | list[BaseMessage] | list[str]) -> int:
                if isinstance(msg, list):
                    if len(msg) > 0 and isinstance(msg[0], str):
                        return count(Procedure("", "", msg).format_steps())

                    return r.model.get_num_tokens_from_messages(msg)

                return r.model.get_num_tokens(msg)

            f.write(f"used {count(prompt)} input tokens and {count(r.completion)} output tokens\n")

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
