import argparse
import asyncio
import glob
import logging
import sys
from os import PathLike
from pathlib import Path

from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage

from utils import spread_gather


# counts for the examples in prompts/
_num_examples = 2
_num_few_shot = 1


def get_prompt_messages(few_shot_as_ex: bool) -> list[BaseMessage]:
    """Load the prompts from text files."""
    module_dir = Path(__file__).resolve().parent
    prompt_dir = module_dir / "prompts"

    # load task and instructions
    with (prompt_dir / "context.txt").open() as f:
        sys_msg = f.read().strip()
    # load examples
    sys_msg += "\n\nHere are a few examples of the expected output:"
    for ex in range(1, _num_examples + 1):
        with (prompt_dir / "examples" / f"{ex}.txt").open() as f:
            sys_msg += f"\n\nBEGIN EXAMPLE {ex}\n" + f.read().strip() + f"\nEND EXAMPLE {ex}"
    sys_msg = sys_msg.strip()

    out = [SystemMessage(content=sys_msg)]

    # load few-shot examples
    for ex in range(1, _num_few_shot + 1):
        # If we're using the few-shot pairs as examples, we just need the example output.
        if not few_shot_as_ex:
            with (prompt_dir / "few_shot" / f"{ex}_input.txt").open() as f:
                out.append(HumanMessage(content=f.read().strip()))
        with (prompt_dir / "few_shot" / f"{ex}_output.txt").open() as f:
            if few_shot_as_ex:
                add = "\n\nBEGIN EXAMPLE {exnum}\n{content}\nEND EXAMPLE {exnum}".format(
                    content=f.read().strip(),
                    exnum=ex + _num_examples,
                )
                out[0].content += add
            else:
                out.append(AIMessage(content=f.read().strip()))

    return out


def format_output(s: str) -> str:
    """Strip out BEGIN EXAMPLE and END EXAMPLE if present, and ensure the output ends with a
    newline."""
    # ensure it ends with a newline
    if s and s[-1] != "\n":
        s += "\n"

    l_idx = s.find("BEGIN EXAMPLE")
    if l_idx == -1:
        return s
    # move to next line
    l_idx = s.index("\n", l_idx) + 1

    r_idx = s.rfind("END EXAMPLE")
    if r_idx == -1:
        return s[l_idx:]

    return s[l_idx:r_idx]  # keep the newline before END EXAMPLE


async def format_procedure(
    model: BaseChatModel,
    prompt: list[BaseMessage],
    filepath: str | PathLike,
    logger: logging.Logger,
):
    """Format a single procedure with GPT-4 and store it in formatted/."""
    filepath = Path(filepath)
    logger.debug("summarizing '%s'", filepath)

    messages = prompt.copy()
    with filepath.open(encoding="utf-8") as f:
        messages.append(HumanMessage(content=f.read().strip()))
    tokens = model.get_num_tokens_from_messages(messages)

    token_limit = 8192
    if tokens >= token_limit - 200:
        logger.info("prompt is too long for '%s': %d >= %d", filepath, tokens, token_limit - 200)
        return

    dest = Path(str(filepath).replace("full", "formatted"))
    if dest == filepath:
        raise ValueError("path '%s' not in full/; exiting to avoid overwriting", filepath)

    resp = await model.agenerate(messages=[messages])
    procedure = resp.generations[0][0].text

    try:
        procedure = format_output(procedure)
    except ValueError:
        logger.exception("error formatting '%s'; saving for manual inspection", dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as file:
        file.write(procedure)


def count_tokens(model: BaseChatModel, path: str) -> int:
    """Counts the tokens of all markdown files in a directory"""
    total_tokens = 0
    for filepath in glob.iglob(path + "/**/*.md", recursive=True):
        with Path(filepath).open(encoding="utf-8") as file:
            content = file.read()
            total_tokens += model.get_num_tokens(content)
    return total_tokens


def count_markdown_files(model: BaseChatModel, path: str, limit: int = 0) -> int:
    """Counts markdown files in a directory that have more than `limit` tokens"""
    out = 0
    for filepath in glob.iglob(path + "/**/*.md", recursive=True):
        with Path(filepath).open(encoding="utf-8") as file:
            content = file.read()
            if model.get_num_tokens(content) > limit:
                out += 1
    return out


def make_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("main")
    logger.addHandler(logging.StreamHandler(sys.stderr))
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    return logger


async def format_docs(
    path: str | PathLike,
    n: int,
    logger: logging.Logger,
    few_shot_as_ex: bool = False,
    n_workers: int = 5,
):
    """Summarize all files in the directory and save in a separate tree under formatted/.

    The files must have the string "full" in their path, which is replaced by "formatted"."""
    if n == 0:
        return

    n_workers = min(n_workers, n)

    prompt = get_prompt_messages(few_shot_as_ex)

    chatgpt = ChatOpenAI(temperature=0.5, model_name="gpt-4-0613")

    path = Path(path)
    files = list(path.glob("**/*.md"))
    if n >= 0 and len(files) > n:
        files = files[:n]

    await spread_gather(
        lambda fp: format_procedure(chatgpt, prompt, fp, logger),
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
        help="path to the dataset; should contain procedures/full/",
    )
    parser.add_argument("-n", type=int, default=-1, help="quit after creating this many procedures")
    parser.add_argument(
        "--few-shot-as-examples",
        action="store_true",
        help=(
            "don't provide any few-shot examples, just paste the few-shot outputs as extra "
            "examples in the system message"
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="show debug messages")

    args = parser.parse_args()

    data_dir = Path(args.data_dir) / "procedures" / "full"
    logger = make_logger(args.verbose)

    asyncio.run(format_docs(data_dir, args.n, logger, args.few_shot_as_examples))
