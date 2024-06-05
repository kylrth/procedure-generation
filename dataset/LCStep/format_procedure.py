import argparse
import asyncio
import hashlib
import logging
import re
import sys
from os import PathLike
from pathlib import Path

from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

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

    out: list[BaseMessage] = [SystemMessage(content=sys_msg)]

    # load few-shot examples
    for ex in range(1, _num_few_shot + 1):
        # If we're using the few-shot pairs as examples, we just need the example output.
        if not few_shot_as_ex:
            with (prompt_dir / "few_shot" / f"{ex}_input.txt").open() as f:
                out.append(HumanMessage(content=f.read().strip()))
        with (prompt_dir / "few_shot" / f"{ex}_output.txt").open() as f:
            if few_shot_as_ex:
                exnum = ex + _num_examples
                add = f"\n\nBEGIN EXAMPLE {exnum}\n{f.read().strip()}\nEND EXAMPLE {exnum}"
                out[0].content += add
            else:
                out.append(AIMessage(content=f.read().strip()))

    return out


_verbose_chain = re.compile(
    r"\n +> Entering new (\w+) chain\.\.\..*?> Finished( \1)? chain\.\n", re.DOTALL
)
_verbose_llm_chain = re.compile(
    r"\n +> Entering new LLMChain chain\.\.\..*?> Finished( LLMChain)? chain\.\n", re.DOTALL
)
_verbose_api_chain = re.compile(
    r"\n +> Entering new API(\w+) chain\.\.\..*?> Finished( API\1)? chain\.\n", re.DOTALL
)
_flare_extra = re.compile(
    r'```\nquery = "explain in great detail.*(```\nflare.run\("how are)', re.DOTALL
)
_local_loading = re.compile(
    r"\n```(?:[^`]|\`(?!``))*?llama\.cpp: loading model from(?:[^`]|\`(?!``))*?```\n", re.DOTALL
)
_llama_timings = re.compile(r"\n\s+llama_print_timings:.*")
_interm_steps_section = re.compile(r"\n## Return Intermediate Steps.*?(?=\n##)", re.DOTALL)
_table_info_section = re.compile(r"\n### Custom Table Info.*?(?=\n###)", re.DOTALL)
_local_models_section = re.compile(r"\n## Using Local Language Models.*", re.DOTALL)
_verbosity_comment = re.compile(r"(verbose ?= ?True.*)")
_checker_examples = re.compile(r"```[^`]*?The Greenland Sea.*", re.DOTALL)
_interm_chain_test = re.compile(
    r"\n```\n# test the intermediate.*> Finished chain\..*?```", re.DOTALL
)
_api_reference = re.compile(r"\n#### API Reference.*?(?=\n##)", re.DOTALL)
_knowledge_base_text = re.compile(r"(Sleep Haven product \d: .*\n.{150}).*(.{150})")
_new_comment = re.compile(r"\n\s+###+ NEW ###+")
_created_chunks = re.compile(r"\n\s+Created a chunk of size \d+, which.*")
_end_convo = re.compile(r"\n```\nsales_agent\.human_step\(\n\s*\"Yes, I am looking.*", re.DOTALL)
_notebook_source = re.compile(r"\nThis notebook was originally published.*\n")
_leverage = re.compile(r"\nWe leverage.*\n")
_schematic = re.compile(r"\nHere is the schematic.*\n\n### Architecture.*\n")
_company_business = re.compile(r" We offer a range of high-quality.*needs of our customers\.")
_company_mission = re.compile(
    r" by providing them with the best possible.*exceptional products and customer service"
)
_only_role = re.compile(
    r"Do not change roles!\s+Do[^\n]*else\.\s+[^\n]*of ([^\n]+)\.\s+Stop[^\n]*", re.DOTALL
)
_short_convo = re.compile(
    r"\s+Bids:\s+Donald Trump bid: 2[^\)]*Warren\): Thank you, but I disagree.*", re.DOTALL
)
_next_speaker = re.compile(
    r"Stop\? False\s+Next speaker: Ronny.*?Aasif.*?(?=Stop\? False)", re.DOTALL
)
_convo_details = re.compile(r"the following topic: The.*?Ohio\.", re.DOTALL)
_printed_descriptions = re.compile(r"\n```\nfor name, description.*?```\n```.*?```", re.DOTALL)
_tic_tac_toe = re.compile(r"## Tic-Tac-Toe.*?(?=##)", re.DOTALL)
_convos = [
    (
        r"\1ing yourself, and clarify why \2",
        re.compile(r"(Start the conversation by introduc).*(you are contacting the prospect\.)"),
    ),
    (
        r"Make sure you're talking to someone who can purchase.",
        re.compile(r"Qualify the prospect by.*to make purchasing decisions\."),
    ),
    (
        r"\1 can benefit them.",
        re.compile(r"(Briefly explain how your product).*apart from competitors\."),
    ),
    (r"\1 and \2", re.compile(r"(Ask open-ended questions).*(take notes\.)")),
    (
        "Present a solution based on their needs.",
        re.compile(r"Based on the prospect.*pain points\."),
    ),
    ("Address objections.", re.compile(r"Address any objections.*your claims\.")),
    (
        "Propose a demo or trial, and summarize the discussion.",
        re.compile(r"Ask for the sale.*the benefits\."),
    ),
]


def _convo_repl(text: str) -> str:
    for repl, reg in _convos:
        text = reg.sub(repl, text)

    return text


def _empty_repl(text: str, *regs) -> str:
    for reg in regs:
        text = reg.sub("", text)

    return text


def _replace_except_first(new_text: str):
    def _replace(match):
        _replace.count += 1
        if _replace.count == 1:
            return match.group(0)
        return new_text

    # init count attribute
    _replace.count = 0

    return _replace


# These are the files whose text needs to be modified. We check the hash of the text to make
# sure the text hasn't changed since the modification was designed.
input_modifications = {
    Path("docs/procedures/full/docs/modules/agents/how_to/chatgpt_clone.md"): (
        "7db02015cdd6b79dc270e0eeda9168c2",
        lambda text: _verbose_llm_chain.sub("", text.replace("verbose=True", "verbose=False", 1)),
    ),
    Path("docs/procedures/full/docs/use_cases/question_answering/how_to/flare.md"): (
        "f8b6bc54313c74477a0e57f3adccf6ac",
        lambda text: _flare_extra.sub(r"\1", text, 1),
    ),
    Path("docs/procedures/full/docs/use_cases/question_answering/how_to/local_retrieval_qa.md"): (
        "70e6bbe66b446fc9041c8630cba903b0",
        lambda text: _llama_timings.sub(
            "", _local_loading.sub("", text.replace("verbose=True", "verbose=False", 1))
        ),
    ),
    Path("docs/procedures/full/docs/use_cases/chatbots/voice_assistant.md"): (
        "d647ea2013378392503a68af13e52af7",
        lambda text: _verbose_llm_chain.sub("", text.replace("verbose=True", "verbose=False", 1)),
    ),
    Path("docs/procedures/full/docs/use_cases/tabular/sqlite.md"): (
        "54d1ccdf733ee35c31be0e7f0679de50",
        lambda text: _empty_repl(
            text, _interm_steps_section, _table_info_section, _local_models_section
        ),
    ),
    Path("docs/procedures/full/docs/use_cases/apis/openapi.md"): (
        "3a261378e7698b7dea0788cd82e429c6",
        lambda text: _verbose_api_chain.sub(
            "",
            _verbosity_comment.sub(
                r"\1  # some output is redacted from this tutorial for brevity", text
            ),
        ),
    ),
    Path("docs/procedures/full/docs/use_cases/agents/sales_agent_with_context.md"): (
        "af1f52f5916c656db50263eedc399176",
        lambda text: _convo_repl(
            _empty_repl(
                _knowledge_base_text.sub(
                    r"\1<SHORTENED FOR BREVITY>\2",
                    _api_reference.sub(
                        "",
                        _interm_chain_test.sub("", text),
                    ),
                ),
                _company_business,
                _company_mission,
                _schematic,
                _leverage,
                _notebook_source,
                _end_convo,
                _created_chunks,
                _new_comment,
            ),
        ),
    ),
    Path("docs/procedures/full/docs/use_cases/self_check/llm_summarization_checker.md"): (
        "8c5122794a3abb7dac204edbb21bf688",
        lambda text: _checker_examples.sub("", text),
    ),
    Path("docs/procedures/full/docs/use_cases/agent_simulations/multiagent_bidding.md"): (
        "5f1af268607e0fb410861b4ed37bc3ec",
        lambda text: _short_convo.sub("", _only_role.sub(r"Only speak as \1.", text)),
    ),
    Path("docs/procedures/full/docs/use_cases/agent_simulations/multiagent_authoritarian.md"): (
        "f970b70386c686a43f2d385fc904ebb6",
        lambda text: _convo_details.sub(
            _replace_except_first("<SHORTENED FOR BREVITY>"),
            _empty_repl(
                _only_role.sub(r"Only speak as \1.", text),
                _api_reference,
                _next_speaker,
                _printed_descriptions,
            ),
        ),
    ),
    Path("docs/procedures/full/docs/use_cases/agent_simulations/petting_zoo.md"): (
        "fe0502dc2e70c938c5b9650b979d148a",
        lambda text: _tic_tac_toe.sub("", text),
    ),
    Path("docs/procedures/full/docs/use_cases/agent_simulations/two_agent_debate_tools.md"): (
        "f08e35a127ef190fbf6c16ba55a7fd89",
        lambda text: _verbose_chain.sub("", text.replace("verbose=True", "verbose=False")),
    ),
}


def prepare_input_text(filepath: Path, logger: logging.Logger) -> str:
    """Read the file and possibly modify the text to fit in the context window."""
    with filepath.open(encoding="utf-8") as f:
        text = f.read().strip()

    if filepath not in input_modifications:
        return text

    want_h, modify = input_modifications[filepath]
    h = hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()
    if h != want_h:
        logger.info("contents of '%s' have changed, new hash is %s", filepath, h)
        logger.info("continuing with text unchanged")
        return text

    return modify(text)


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
    model: ChatOpenAI,
    prompt: list[BaseMessage],
    filepath: str | PathLike,
    logger: logging.Logger,
):
    """Format a single procedure with GPT-4 and store it in formatted/."""
    filepath = Path(filepath)
    logger.debug("summarizing '%s'", filepath)

    messages = prompt.copy()
    messages.append(HumanMessage(content=prepare_input_text(filepath, logger)))
    tokens = model.get_num_tokens_from_messages(messages)

    token_limit = 8192
    if tokens >= token_limit - 600:
        # We've seen outputs as long as 552, so we need to leave at least that much room.
        logger.info("prompt is too long for '%s': %d >= %d", filepath, tokens, token_limit - 600)
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

    logger.debug("stored '%s'", dest)


def count_tokens(model: ChatOpenAI, path: str) -> int:
    """Counts the tokens of all markdown files in a directory"""
    total_tokens = 0
    for file in Path(path).glob("**/*.md"):
        with file.open(encoding="utf-8") as f:
            content = f.read()
            total_tokens += model.get_num_tokens(content)
    return total_tokens


def count_markdown_files(model: ChatOpenAI, path: str, limit: int = 0) -> int:
    """Counts markdown files in a directory that have more than `limit` tokens"""
    out = 0
    for file in Path(path).glob("**/*.md"):
        with file.open(encoding="utf-8") as f:
            content = f.read()
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

    prompt = get_prompt_messages(few_shot_as_ex)

    chatgpt = ChatOpenAI(temperature=0.5, model="gpt-4-0613")

    path = Path(path)
    files = list(path.glob("**/*.md"))
    if n >= 0 and len(files) > n:
        files = files[:n]

    n_workers = min(n_workers, len(files))
    logger.debug(f"{len(files)} files, {n_workers} workers")

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
        help="path to the dataset directory, which should contain procedures/full/",
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
