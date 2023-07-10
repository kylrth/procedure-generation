# ruff: noqa: T201
# This script needs to print.
import argparse
import asyncio
import glob
import json
import logging
from pathlib import Path

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from run import make_logger


chatgpt = ChatOpenAI(temperature=0, model_name="gpt-4")
with Path("./prompt.json").open() as f:
    prompt = json.load(f)
context = prompt["context"]
task = prompt["task"]
TOKEN__LIMIT = 8192


async def format_procedure(filepath: str, logger: logging.Logger) -> None:
    """Formats a single procedure with GPT-4"""
    with Path.open(Path(filepath), encoding="utf-8") as file:
        full_procedure = file.read()
        des = filepath.replace("full", "formatted")
    messages = [SystemMessage(content=context), HumanMessage(content=task.format(full_procedure))]
    if chatgpt.get_num_tokens_from_messages(messages) < TOKEN__LIMIT:
        resp = await chatgpt.agenerate(messages=[messages])
        formatted_proc = resp.generations[0][0].text
        Path.mkdir(Path(des), parents=True, exist_ok=True)
        with Path.open(Path(des), "w") as file:
            file.write(formatted_proc)
    else:
        logger.debug(filepath)


def count_tokens(path: str) -> int:
    """Counts the tokens of all markdown files in a directory"""
    total_tokens = 0
    for filepath in glob.iglob(path + "/**/*.md", recursive=True):
        with Path.open(Path(filepath), encoding="utf-8") as file:
            content = file.read()
            total_tokens += chatgpt.get_num_tokens(content)
    return total_tokens


def count_markdown_files(directory: str) -> int:
    """Counts markdown files in a directory"""
    return len(glob.glob(directory + "/**/*.md", recursive=True))


async def worker(queue):
    while True:
        t = await queue.get()
        try:
            await t
        finally:
            queue.task_done()


async def format_docs(path: str, n_workers: int = 5) -> None:
    """Formats all files in the directory and saves them to a different dir"""
    logger = make_logger("exceeded_token_limit")
    queue = asyncio.Queue()
    workers = [asyncio.create_task(worker(queue)) for _ in range(n_workers)]
    for filepath in glob.iglob(path + "/**/*.md", recursive=True):
        await queue.put(format_procedure(filepath, logger))
    await queue.join()
    for w in workers:
        w.cancel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data-dir",
        type=str,
        default="./docs",
        help="directory containing the scraped dataset",
    )
    args = parser.parse_args()
    asyncio.run(format_docs(args.data_dir))
