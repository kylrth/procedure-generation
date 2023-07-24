import argparse
import asyncio
import logging
import sys
from os import PathLike
from pathlib import Path

from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

from utils import spread_gather


def get_prompt_messages() -> list[BaseMessage]:
    module_dir = Path(__file__).resolve().parent
    prompt_dir = module_dir / "prompts"

    with (prompt_dir / "check.txt").open() as f:
        sys_msg = f.read().strip()
    with (prompt_dir / "context.txt").open() as f:
        sys_msg.format(instructions=f.read().strip())

    return [SystemMessage(content=sys_msg)]


def make_logger() -> logging.Logger:
    logger = logging.getLogger("main")
    logger.addHandler(logging.StreamHandler(sys.stderr))
    logger.setLevel(logging.DEBUG)

    return logger


async def judge_doc(
    model: BaseChatModel,
    prompt: list[BaseMessage],
    formatted_path: str | PathLike,
    logger: logging.Logger,
):
    full_path = Path(str(formatted_path).replace("formatted", "full"))

    messages = prompt.copy()
    with Path(formatted_path).open() as f:
        messages.append(HumanMessage(content=f.read().strip()))

    # TODO:
    # - judge with model
    # - judge by searching text for verbosity, model temp
    #   - should this be a demerit or should this info just get included in the prompt for the model
    #     to decide?


async def check_docs(root: str | PathLike, n: int, n_workers: int = 5):
    if n == 0:
        return

    logger = make_logger()
    n_workers = min(n_workers, n)

    prompt = get_prompt_messages()

    chatgpt = ChatOpenAI(temperature=0.5, model_name="gpt-4-0613")

    root = Path(root)

    files = list((root / "formatted").glob("**/*.md"))
    if n >= 0 and len(files) > n:
        files = files[:n]

    await spread_gather(
        lambda fp: judge_doc(chatgpt, prompt, fp, logger),
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
