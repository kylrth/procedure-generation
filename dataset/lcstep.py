"""Tools for loading the LCStep dataset. For creating the dataset, see the LCStep/ folder."""

import bisect
import pickle
import re
from os import PathLike
from pathlib import Path
from typing import cast

from .base import Dataset, Doc, GraphProcedure, LinearProcedure


def load_api_ref(data_dir: str | PathLike = "./dataset/") -> list[Doc]:
    """Returns the API reference docs."""
    root = Path(data_dir) / "LCStep" / "docs" / "api"

    # walk the docs to collect all markdown files
    out = []
    for file in root.glob("**/*.md"):
        api_fqn = file.name[:-8]  # remove .html.md
        ref = file.read_text()

        # sort by API name
        bisect.insort(out, Doc(title=api_fqn, contents=ref), key=lambda d: d.title)

    return out


def load_concept_docs(data_dir: str | PathLike = "./dataset/") -> list[Doc]:
    """Returns the conceptual documentation."""
    root = Path(data_dir) / "LCStep" / "docs" / "concepts"

    # walk the docs to collect all markdown files
    out = []
    for file in root.glob("**/*.md"):
        path = str(file)[:-3]  # remove .md
        ref = file.read_text()

        # sort by path
        bisect.insort(out, Doc(title=path, contents=ref), key=lambda d: d.title)

    return out


_ordering_re = re.compile(r"\d+\. ")


def procedure_from_text(text: str) -> tuple[LinearProcedure, str]:
    """Parse text containing a formatted LCStep procedure.

    The procedure is returned along with the side note."""
    chunks = text.split("\n\n")
    if len(chunks) < 2 or len(chunks) > 3:  # goal + steps + optional side note
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
    if len(chunks) == 3:  # side note appears as 3rd chunk, if any
        side_note = chunks[2].strip()
        _prefixes = ["Side note: ", "Note: "]
        for p in _prefixes:
            if side_note.startswith(p):
                side_note = side_note[len(p) :].strip()  # remove final newline
                break
        else:
            raise ValueError(f"procedure side note not marked with '{_prefixes[0]}'")

    return LinearProcedure(resources, goal, steps), side_note


def _goal_and_resources_from_text(text: str) -> tuple[str, str]:
    chunks = text.split("\n")
    if len(chunks) < 1:
        raise ValueError("top of procedure does not contain goal and resources lines")
    if len(chunks) == 1:
        goal = chunks[0]
        resources = "Resources: None"
    else:
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


def load_formatted_docs(
    data_dir: str | PathLike = "./dataset/",
) -> list[dict[str, str | LinearProcedure]]:
    """Returns the API reference docs.

    The returned dicts have keys "path": str, "procedure": Procedure, and "side_note": str.
    """
    root = Path(data_dir) / "LCStep" / "docs" / "procedures" / "formatted"

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

            # sort by length of step text, as a proxy for difficulty
            bisect.insort(
                dicts,
                {
                    "path": path,
                    "procedure": p,
                    "side_note": side_note,
                },
                key=lambda v: len("\n".join(v["procedure"].steps)),
            )

    for i, d in enumerate(dicts):
        d["id"] = i

    return dicts


class LCStep(Dataset):
    def _init_procedures(self) -> list[LinearProcedure]:
        return [cast(LinearProcedure, d["procedure"]) for d in load_formatted_docs(self.dir)]

    def _init_graphs(self) -> list[GraphProcedure]:
        d = self.dir / "graphs" / "lcstep"
        file_list = d.glob("*.pkl")
        graph_list = []
        for file in file_list:
            with file.open("rb") as f:
                graph = pickle.load(f)
            graph_list.append(graph)

        # this graph is malformed; skip
        del graph_list[32]

        return graph_list

    def _get_docs(self) -> list[Doc]:
        api_ref = load_api_ref(self.dir)
        concept_docs = load_concept_docs(self.dir)

        return api_ref + concept_docs


if __name__ == "__main__":
    # ruff: noqa: T201

    ds = load_api_ref()
    print("API ref dataset:", len(ds))
    print("example:", repr(ds[0])[:120] + "...")

    ds = load_concept_docs()
    print("concept docs dataset:", len(ds))
    print("example:", repr(ds[0])[:120] + "...")

    ds = load_formatted_docs()
    print("formatted docs dataset:", len(ds))
    print("example:", repr(ds[0]))
