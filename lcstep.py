"""tools for the LCStep dataset"""

import bisect
import re
from os import PathLike
from pathlib import Path

from datasets import Dataset


def load_api_ref(data_dir: str | PathLike = "./dataset/docs") -> Dataset:
    root = Path(data_dir) / "api"

    # walk the docs to collect all markdown files
    dicts = []
    for file in root.glob("**/*.md"):
        api_fqn = file.name[:-8]  # remove .html.md
        ref = file.read_text()

        # sort by API name
        bisect.insort(dicts, {"api": api_fqn, "ref": ref}, key=lambda v: v["api"])

    return Dataset.from_list(dicts)


def load_concept_docs(data_dir: str | PathLike = "./dataset/docs") -> Dataset:
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


def procedure_from_text(text: str) -> tuple[str, list[str], str]:
    """Parse text containing a formatted procedure."""
    chunks = text.split("\n\n")
    if len(chunks) < 2 or len(chunks) > 3:  # goal + steps + optional side note  # noqa: PLR2004
        raise ValueError("procedure does not contain 2-3 chunks")

    # parse the goal
    goal = chunks[0]
    _prefix = "Goal: "
    if not goal.startswith(_prefix):
        raise ValueError(f"procedure goal not marked with '{_prefix}'")
    goal = goal[len(_prefix) :]

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
    side_note = chunks[2].strip() if len(chunks) == 3 else ""  # noqa: PLR2004
    _prefix = "side note: "
    if side_note:
        if not side_note.lower().startswith(_prefix):
            raise ValueError(f"procedure side note not marked with '{_prefix}'")
        side_note = side_note[len(_prefix) :].strip()  # remove final newline

    return goal, steps, side_note


def text_from_procedure(goal: str, steps: list[str], side_note: str) -> str:
    """Format the text of a procedure. `goal` and `side_note` are optional and ignored if empty."""
    out = "Goal: " + goal + "\n\n" if goal else ""  # allow blank goal if formatting just steps

    for i, step in enumerate(steps):
        out += f"{i+1}. {step}\n"

    if side_note:
        out += "\nSide note: " + side_note + "\n"

    return out


def load_formatted_docs(data_dir: str | PathLike = "./dataset/docs") -> Dataset:
    root = Path(data_dir) / "procedures" / "formatted"

    # walk the docs to collect all markdown files
    dicts = []
    for file in root.glob("**/*.md"):
        path = str(file)[:-3]  # remove .md
        full_text = file.read_text()

        for ref in full_text.split("\nNEW PROCEDURE\n"):
            text = ref.strip()
            try:
                goal, steps, side_note = procedure_from_text(text)
            except ValueError as e:
                raise ValueError(f"could not process '{file}'") from e

            # sort by length of text, as a proxy for difficulty
            bisect.insort(
                dicts,
                {"path": path, "ref": text, "goal": goal, "steps": steps, "side_note": side_note},
                key=lambda v: len(v["ref"]),
            )

    return Dataset.from_list(dicts)


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
