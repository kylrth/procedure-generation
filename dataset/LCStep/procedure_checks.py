# ruff: noqa: T201
# This script needs to print.

import argparse
import asyncio
import logging
import os
import subprocess
import sys
import tempfile
from difflib import Differ
from os import PathLike
from pathlib import Path
from typing import IO

from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

from utils import spread_gather


class Recorder:
    """Keep track of which files we have already reviewed, so that we can pick up where we left off
    if we exit and rerun the script."""

    path: Path
    f: IO[str] | None
    done: set[str]

    def __init__(self, path: str | PathLike):
        self.path = Path(path)
        self.f = None

    def __enter__(self):
        self.f = self.path.open("a+")
        self.f.seek(0)
        self.done = {line.strip() for line in self.f.readlines()}

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.f is not None:
            self.f.close()

    def __contains__(self, obj):
        if self.f is None:
            raise ValueError("inclusion check called outside of context")

        return str(obj) in self.done

    def add(self, file: str | PathLike):
        if self.f is None:
            raise ValueError("add method called outside of context")

        file = str(file)

        self.done.add(file)
        self.f.write(file + "\n")


def get_prompt_messages() -> list[BaseMessage]:
    module_dir = Path(__file__).resolve().parent
    prompt_dir = module_dir / "prompts"

    with (prompt_dir / "check.txt").open() as f:
        sys_msg = f.read().strip()
    with (prompt_dir / "context.txt").open() as f:
        sys_msg.format(instructions=f.read().strip())

    return [SystemMessage(content=sys_msg)]


def make_logger(name: str) -> logging.Logger:
    log_dir = Path("checks_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"

    logger = logging.getLogger(name)
    logger.propagate = False
    logger.addHandler(logging.FileHandler(log_file, "w"))
    logger.setLevel(logging.DEBUG)
    return logger


_editor = os.environ.get("VISUAL") or os.environ.get("EDITOR", "nano")


def edit_text(text: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".tmp", mode="w+") as f:
        f.write(text)
        f.flush()

        subprocess.call([_editor, f.name])

        f.seek(0)
        return f.read()


def diff(old: str, new: str) -> str:
    d = Differ()

    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)

    return "".join(
        line[2:] if line.startswith("  ") else line for line in d.compare(old_lines, new_lines)
    )


async def judge_doc(
    model: BaseChatModel,
    prompt: list[BaseMessage],
    formatted_path: str | PathLike,
    rec: Recorder,
):
    if "fixed" in str(formatted_path):
        return

    formatted_path = Path(formatted_path)

    if formatted_path in rec:
        print(f"skipped '{formatted_path}'", file=sys.stderr)

        return

    messages = prompt.copy()
    with formatted_path.open() as f:
        text = f.read().strip()
    messages.append(HumanMessage(content=text))

    resp = await model.agenerate(messages=[messages])
    judgement = resp.generations[0][0].text

    if "[FAIL]" not in judgement:
        print(f"file '{formatted_path}' passed", file=sys.stderr)

        rec.add(formatted_path)

        return

    logger = make_logger(formatted_path.stem)

    print(f"file '{formatted_path}':", file=sys.stderr)
    logger.debug("file: %s", formatted_path)
    logger.debug("judgement: %s", judgement)

    if "Goal:" in judgement:
        criteria = judgement[: judgement.index("Goal:")].strip()
        new_text = judgement[judgement.index("Goal:") :]

        print("  criteria:\n" + criteria)
        input("  press enter to edit the diff (leave blank to abort)")

        modified = edit_text(diff(text, new_text))
        if modified.strip() == "":
            raise ValueError("user aborted by saving empty text")
        with formatted_path.open("w") as f:
            f.write(modified)
        logger.debug("modified: %s", modified)

        rec.add(formatted_path)
        print("  done.")

        return

    print("  could not parse updated procedure from judgement", file=sys.stderr)
    print("  judgement text:\n\n" + judgement, file=sys.stderr)
    input("  press enter to edit the procedure manually")
    subprocess.call([_editor, str(formatted_path)])

    rec.add(formatted_path)
    print("  done.")


async def check_docs(root: str | PathLike, n: int, n_workers: int = 5):
    if n == 0:
        return

    prompt = get_prompt_messages()

    chatgpt = ChatOpenAI(temperature=0.5, model="gpt-4-0613")

    root = Path(root)

    files = list((root / "formatted").glob("**/*.md"))
    if n >= 0 and len(files) > n:
        files = files[:n]

    n_workers = min(n_workers, len(files))

    with Recorder("checks_logs/done.txt") as rec:
        await spread_gather(
            lambda fp: judge_doc(chatgpt, prompt, fp, rec),
            files,
            n_workers,
            len(files),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data-dir",
        type=str,
        default="./docs",
        help="path to the dataset; should contain procedures/formatted/",
    )
    parser.add_argument("-n", type=int, default=-1, help="quit after checking this many procedures")

    args = parser.parse_args()

    data_dir = Path(args.data_dir) / "procedures"
    asyncio.run(check_docs(data_dir, args.n))
