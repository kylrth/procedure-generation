import argparse
import asyncio
import logging
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


def make_logger(name: str) -> logging.Logger:
    log_dir = Path("checks_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"

    logger = logging.getLogger(name)
    logger.propagate = False
    logger.addHandler(logging.FileHandler(log_file, "w"))
    logger.setLevel(logging.DEBUG)
    return logger


async def judge_doc(
    model: BaseChatModel,
    prompt: list[BaseMessage],
    formatted_path: str | PathLike,
):
    if "fixed" in str(formatted_path):
        return

    formatted_path = Path(formatted_path)

    messages = prompt.copy()
    with formatted_path.open() as f:
        messages.append(HumanMessage(content=f.read().strip()))

    resp = await model.agenerate(messages=[messages])
    judgement = resp.generations[0][0].text
    if "[FAIL]" in judgement:
        stem = formatted_path.stem
        fixed_path = formatted_path.parent / (stem + "_fixed" + formatted_path.suffix)
        logger = make_logger(stem)
        logger.debug(judgement)
        with fixed_path.open("w") as f:
            if "Goal:" in judgement:
                f.write(judgement[judgement.index("Goal:") :])
            else:
                f.write(judgement)


async def check_docs(root: str | PathLike, n: int, n_workers: int = 5):
    if n == 0:
        return

    prompt = get_prompt_messages()

    chatgpt = ChatOpenAI(temperature=0.5, model_name="gpt-4-0613")

    root = Path(root)

    files = list((root / "formatted").glob("**/*.md"))
    if n >= 0 and len(files) > n:
        files = files[:n]

    n_workers = min(n_workers, len(files))

    await spread_gather(
        lambda fp: judge_doc(chatgpt, prompt, fp),
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
