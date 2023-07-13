# ruff: noqa: T201
# This script needs to print.
import argparse
import asyncio
import glob
import logging
from pathlib import Path

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


chatgpt = ChatOpenAI(temperature=0, model_name="gpt-4")
TOKEN__LIMIT = 8192

prompt_dir = Path("./prompts")
with (prompt_dir / "context.txt").open() as f:
    context = f.read()
for file in sorted((prompt_dir / "few_shot").glob("*.txt")):
    with file.open() as f:
        context += f.read()
with (prompt_dir / "task.txt").open() as f:
    task = f.read().strip()


def make_logger(name: str) -> logging.Logger:
    """Create a new logger that writes to logs/{name}.log (and nowhere else)."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"

    logger = logging.getLogger(name)
    logger.propagate = False
    logger.addHandler(logging.FileHandler(log_file, "w"))
    logger.setLevel(logging.DEBUG)
    return logger


async def format_procedure(filepath: str, logger: logging.Logger) -> None:
    """Formats a single procedure with GPT-4"""
    print("Working on ", filepath)
    with Path(filepath).open(encoding="utf-8") as file:
        full_procedure = file.read()
        des = filepath.replace("full", "formatted")
    messages = [SystemMessage(content=context), HumanMessage(content=task.format(full_procedure))]
    if chatgpt.get_num_tokens_from_messages(messages) < TOKEN__LIMIT:
        resp = await chatgpt.agenerate(messages=[messages])
        formatted_proc = resp.generations[0][0].text
        Path(des).parent.mkdir(parents=True, exist_ok=True)
        with Path(des).open("w") as file:
            file.write(formatted_proc)
    else:
        logger.debug(filepath)


def count_tokens(path: str) -> int:
    """Counts the tokens of all markdown files in a directory"""
    total_tokens = 0
    for filepath in glob.iglob(path + "/**/*.md", recursive=True):
        with Path(filepath).open(encoding="utf-8") as file:
            content = file.read()
            total_tokens += chatgpt.get_num_tokens(content)
    return total_tokens


def count_markdown_files(path: str, limit: int = 0) -> int:
    """Counts markdown files in a directory that have more than `limit` tokens"""
    out = 0
    for filepath in glob.iglob(path + "/**/*.md", recursive=True):
        with Path(filepath).open(encoding="utf-8") as file:
            content = file.read()
            if chatgpt.get_num_tokens(content) > limit:
                out += 1
    return out


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
