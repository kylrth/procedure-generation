"""Tools for loading the LCStep dataset. For creating the dataset, see the LCStep/ folder."""

import bisect
import re
from os import PathLike
from pathlib import Path

from datasets import Dataset

from dataset.base import Procedure


def load_api_ref(data_dir: str | PathLike = "./dataset/LCStep/docs") -> Dataset:
    root = Path(data_dir) / "api"

    # walk the docs to collect all markdown files
    dicts = []
    for file in root.glob("**/*.md"):
        api_fqn = file.name[:-8]  # remove .html.md
        ref = file.read_text()

        # sort by API name
        bisect.insort(dicts, {"api": api_fqn, "ref": ref}, key=lambda v: v["api"])

    return Dataset.from_list(dicts)


def load_concept_docs(data_dir: str | PathLike = "./dataset/LCStep/docs") -> Dataset:
    root = Path(data_dir) / "concepts"

    # walk the docs to collect all markdown files
    dicts = []
    for file in root.glob("**/*.md"):
        path = str(file)[:-3]  # remove .md
        ref = file.read_text()

        # sort by path
        bisect.insort(dicts, {"path": path, "ref": ref}, key=lambda v: v["path"])

    return Dataset.from_list(dicts)


_ordering_re = re.compile(r"\d+\. ")


def procedure_from_text(text: str) -> tuple[Procedure, str]:
    """Parse text containing a formatted LCStep procedure.

    The procedure is returned along with the side note."""
    chunks = text.split("\n\n")
    if len(chunks) < 2 or len(chunks) > 3:  # goal + steps + optional side note  # noqa: PLR2004
        raise ValueError("procedure does not contain 2-3 chunks")

    goal, resources = _goal_and_resources_from_text(chunks[0])

    # parse the steps (which may be multi-line)
    lines = chunks[1].strip().split("\n")
    steps = []
    for line in lines:
        _prefix = f"{len(steps)+1}. "
        if line.startswith(_prefix):
            steps.append(line[len(_prefix) :])
            continue

        if _ordering_re.match(line):
            raise ValueError("incorrect step order")

        if len(steps) == 0:
            raise ValueError("steps did not start with 1")
        steps[-1] += "\n" + line

    # parse the side note
    side_note = ""
    if len(chunks) == 3:  # noqa: PLR2004  # side note appears as 3rd chunk, if any
        side_note = chunks[2].strip()
        _prefixes = ["Side note: ", "Note: "]
        for p in _prefixes:
            if side_note.startswith(p):
                side_note = side_note[len(p) :].strip()  # remove final newline
                break
        else:
            raise ValueError(f"procedure side note not marked with '{_prefixes[0]}'")

    return Procedure(resources, goal, steps), side_note


def _goal_and_resources_from_text(text: str) -> tuple[str, str]:
    chunks = text.split("\n")
    if len(chunks) != 2:  # noqa: PLR2004  # one line each for goal and resources
        raise ValueError("top of procedure does not contain goal and resources lines")

    goal, resources = chunks

    _prefix = "Goal: "
    if not goal.startswith(_prefix):
        raise ValueError(f"procedure goal not marked with '{_prefix}'")
    goal = goal[len(_prefix) :]

    _prefix = "Resources: "
    if not resources.startswith(_prefix):
        raise ValueError(f"procedure resources not marked with '{_prefix}'")
    resources = resources[len(_prefix) :]

    return goal, resources


def format_procedure(p: Procedure, side_note: str = "") -> str:
    """Format the text of a procedure.

    `goal` and `side_note` are optional and ignored if empty.
    """
    # allow blank goal if formatting just steps
    out = "Goal: " + p.output + "\n\n" if p.output else ""

    out += format_steps(p.steps)

    if side_note:
        out += "\nSide note: " + side_note + "\n"

    return out


def format_steps(steps: list[str]) -> str:
    """Format just the steps of the procedure."""
    out = ""

    for i, step in enumerate(steps):
        out += f"{i+1}. {step}\n"

    return out


def load_formatted_docs(data_dir: str | PathLike = "./dataset/LCStep/docs") -> Dataset:
    root = Path(data_dir) / "procedures" / "formatted"

    # walk the docs to collect all markdown files
    dicts = []
    for file in root.glob("**/*.md"):
        path = str(file)[:-3]  # remove .md
        full_text = file.read_text()

        for ref in full_text.split("NEW PROCEDURE\n"):
            text = ref.strip()
            if not text:
                continue

            try:
                p, side_note = procedure_from_text(text)
            except ValueError as e:
                raise ValueError(f"could not process '{file}'") from e

            # sort by length of text, as a proxy for difficulty
            bisect.insort(
                dicts,
                {
                    "path": path,
                    "ref": text,
                    "input": p._input,
                    "output": p.output,
                    "steps": p.steps,
                    "side_note": side_note,
                },
                key=lambda v: len(v["ref"]),
            )

    ds = Dataset.from_list(dicts)

    # add question IDs
    def add_id(example, idx):
        example["id"] = idx
        return example

    ds = ds.map(add_id, with_indices=True)

    return ds


if __name__ == "__main__":
    # ruff: noqa: T201

    ds = load_api_ref()
    print("API ref dataset:")
    print(ds)
    print("example:", repr(ds[0])[:120] + "...")

    ds = load_concept_docs()
    print("concept docs dataset:")
    print(ds)
    print("example:", repr(ds[0])[:120] + "...")

    ds = load_formatted_docs()
    print("formatted docs dataset:")
    print(ds)
    print("example:", repr(ds[0]))
