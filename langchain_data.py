"""Tools for the LangChain data"""

import bisect
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


def load_formatted_docs(data_dir: str | PathLike = "./dataset/docs") -> Dataset:
    root = Path(data_dir) / "procedures" / "formatted"

    # walk the docs to collect all markdown files
    dicts = []
    for file in root.glob("**/*.md"):
        path = str(file)[:-3]  # remove .md
        ref = file.read_text()

        # TODO get title, break up steps, etc.

        # sort by difficulty
        bisect.insort(dicts, {"path": path, "ref": ref}, key=lambda v: v["path"])

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
    print("example:", repr(ds[0])[:120] + "...")
