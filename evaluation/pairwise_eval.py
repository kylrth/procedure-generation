import argparse
import asyncio
import csv
import logging
import random
import re
import sys
from pathlib import Path
from typing import Sequence

import tqdm
from langchain.schema import HumanMessage, SystemMessage

from dataset import Procedure, format_steps
from evaluation.eval import read_outputs_csv
from systems import Model
from utils import spread_gather


random.seed(1)


def prepare_prompt(proc1_steps: str, proc2_steps: str, gold: Procedure, use_gt: bool):
    if not use_gt:
        prompt = [
            SystemMessage(
                content=(
                    "[Instruction]\n"
                    "Please act as an impartial judge and evaluate the quality of the two answer "
                    "procedures provided below to achieve the same user goal and choose one out of "
                    "the two as the preferred procedure.\n"
                    "For this evaluation, you should primarily evaluate if the procedure achieves "
                    "the user goal adequately or not and whether it uses all the resources "
                    "mentioned in user goal or not. Do not penalize a procedure based on its "
                    "sentences' structure, grammar and wording, but focus on the facts and "
                    "suggestions being made by the procedure; if the procedure leads to the "
                    "specified user goal with sufficient details and is easy to understand on each "
                    "intermediate step, then that procedure should be chosen.\n\n"
                    "Compare the two procedures below and then provide a short explanation of your "
                    "reason for the choice between the two. Be as objective as possible. Based on "
                    "your explanation, you must choose one procedure following this format: "
                    '"[[choice]]" where choice can be 1 or 2 only, for example: "Choice: [[2]]". '
                    "Print this choice at the END only.\n\n"
                )
            ),
            HumanMessage(
                content=(
                    "[User Goal]\n"
                    f"{gold.output} using {gold.input_}\n\n"
                    f"[BEGIN PROCEDURE 1]\n{proc1_steps}\n[END OF PROCEDURE 1]\n\n"
                    f"[BEGIN PROCEDURE 2]\n{proc2_steps}\n[END OF PROCEDURE 2]\n\n"
                    f'Your answer should begin with "Here is my analysis of the comparison between '
                    "the two procedures:\n1."
                )
            ),
        ]
    else:
        prompt = [
            SystemMessage(
                content=(
                    "[Instruction]\n"
                    "Please act as an impartial judge and evaluate the quality of the two answer "
                    "procedures provided below to achieve the same user goal and choose one out of "
                    "the two as the preferred procedure. For your reference, a gold procedure is "
                    "given below which accurately achieves the user goal.\n"
                    "For this evaluation, you should primarily evaluate if the procedure achieves "
                    "the user goal adequately or not and whether it uses all the resources "
                    "mentioned in user goal or not. Do not penalize a procedure based on its "
                    "sentences' structure, grammar and wording, but focus on the facts and "
                    "suggestions being made by the procedure; if the procedure resembles closely "
                    "with the gold procedure and leads to the specified user goal with sufficient "
                    "details and is easy to understand on each intermediate step, then that "
                    "procedure should be chosen.\n\n"
                    "Compare the two procedures below and then provide a short explanation of your "
                    "reason for the choice between the two. Be as objective as possible. Based on "
                    "your explanation, you must choose one procedure following this format: "
                    '"[[choice]]" where choice can be 1 or 2 only, for example: "Choice: [[2]]". '
                    "Print this choice at the END only.\n\n"
                )
            ),
            HumanMessage(
                content=(
                    "[User Goal]\n"
                    f"{gold.output} using {gold.input_}\n\n"
                    f"[BEGIN GOLD]\n{gold.format_steps()}\n[END GOLD]\n\n"
                    f"[BEGIN PROCEDURE 1]\n{proc1_steps}\n[END OF PROCEDURE 1]\n\n"
                    f"[BEGIN PROCEDURE 2]\n{proc2_steps}\n[END OF PROCEDURE 2]\n\n"
                    f'Your answer should begin with "Here is my analysis of the comparison between '
                    "the two procedures:\n1."
                )
            ),
        ]

    return prompt


def result_parser(sentence: str) -> int:
    if not sentence:
        return -2
    match = re.findall(
        r"\s*\[\[(\d+)\]\]\s*$", sentence
    )  # Check pattern at end of string [[<score>]]
    if not match:
        match = re.findall(
            r"\s*\[\[\w*:\s*(\d+)\]\]\s*$", sentence
        )  # Check at end of string [[Choice: <score>]]
        if not match:
            match = re.findall(
                r"\s*\[\[(\d+)\]\]\s*", sentence
            )  # Check anywhere in string [[<score>]]
            if not match:
                match = re.findall(
                    r"\s*\[\[\w*:\s*(\d+)\]\]\s*", sentence
                )  # Check anywhere in string [[Choice: <score>]]
                if not match:
                    match = re.search(r"\s*(N\/A)\s*$", sentence)  # Check any form of N/A anywhere
                    if not match:
                        return -3
                    return -1

    return int(match[-1])


async def get_choice(model: Model, prompt, seed: int, if_inv: bool) -> int:
    answer = await model.generate(prompt, seed=seed)
    score = result_parser(answer)
    if if_inv:
        if score == 1:
            score = 2
        elif score == 2:
            score = 1
    return score


def get_final_choice(score: list[int]) -> int:
    score_rag = sum(1 for s in score if s == 1)
    score_aag = sum(1 for s in score if s == 2)

    if score_aag == 0 and score_rag == 0:
        final_choice = -1
    elif score_aag == score_rag:
        final_choice = 0
    elif score_aag > score_rag:
        final_choice = 2
    else:
        final_choice = 1
    return final_choice


async def aevaluate(
    logger: logging.Logger,
    model: Model,
    use_gt: bool,
    num_eval_each_type: int,
    q_id: int,
    gold: Procedure,
    sys1_generated: list[str],
    sys2_generated: list[str],
) -> dict[str, int]:
    seed_list = [random.randint(0, sys.maxsize) for _ in range(2 * num_eval_each_type)]
    prompt = prepare_prompt(
        format_steps(sys1_generated), format_steps(sys2_generated), gold, use_gt
    )
    prompt_inv = prepare_prompt(
        format_steps(sys2_generated), format_steps(sys1_generated), gold, use_gt
    )
    prompt_list = [pt for _ in range(num_eval_each_type) for pt in [prompt, prompt_inv]]
    logger.debug("Prompt prepared")

    scores = await asyncio.gather(
        *[
            get_choice(model, p, s, if_inv=(i % 2 == 1))
            for i, (p, s) in enumerate(zip(prompt_list, seed_list))
        ]
    )
    final_choice = get_final_choice(scores)
    logger.debug(f"LLM choice was {final_choice}")

    return {"question_id": q_id, "choice": final_choice}


def check_ids(one: Sequence[int], two: Sequence[int]):
    if abs(len(one) - len(two) > 10):
        raise ValueError(f"first CSV has {len(one)} elements, second has {len(two)}")

    in_one = set(one)
    in_two = set(two)

    only_in_one = in_one.difference(in_two)
    only_in_two = in_two.difference(in_one)

    if only_in_one and only_in_two:
        raise ValueError(
            f"first CSV is missing {' '.join(str(i) for i in sorted(only_in_two))}, "
            f"second is missing {' '.join(str(i) for i in sorted(only_in_one))}"
        )
    if only_in_two:
        raise ValueError(f"first CSV is missing {' '.join(str(i) for i in sorted(only_in_two))}")
    if only_in_one:
        raise ValueError(f"second CSV is missing {' '.join(str(i) for i in sorted(only_in_one))}")


async def pairwise_eval(
    logger: logging.Logger,
    model: Model,
    csv1: Path,
    csv2: Path,
    out: Path,
    tqdm_desc: str | None = None,
):
    csv1_list = sorted(read_outputs_csv(csv1), key=lambda t: t[0])
    csv2_list = sorted(read_outputs_csv(csv2), key=lambda t: t[0])
    check_ids([i for i, _, _ in csv1_list], [i for i, _, _ in csv2_list])

    out.parent.mkdir(parents=True, exist_ok=True)

    with tqdm.tqdm(total=len(csv1_list), desc=tqdm_desc) as pbar, out.open("w", newline="") as f:
        w = csv.DictWriter(f, ["question_id", "choice"])
        w.writeheader()

        async def _task(s1: tuple[int, Procedure, list[str]], s2: tuple[int, Procedure, list[str]]):
            d = await aevaluate(logger, model, args.gt, args.nruns, *s1, s2[2])
            w.writerow(d)
            pbar.update(1)

        await spread_gather(
            lambda s1_s2: _task(s1_s2[0], s1_s2[1]),
            zip(csv1_list, csv2_list, strict=True),
            args.workers,
        )


def build_combinations(
    logger: logging.Logger,
    datasets: list[str],
    systems: list[str],
    embeddings: list[str],
    *,
    base_system: str | None = None,
    base_embedding: str | None = None,
) -> list[tuple[Path, Path, Path]]:
    if base_system is None and base_embedding is None:
        raise ValueError("neither system- nor embedding-based pair-up specified")

    if base_system is not None:
        if base_embedding is not None:
            raise ValueError("both a system- and an embedding-based pair-up were specified")

        base = base_system
        pivot = systems
    else:
        base = base_embedding
        pivot = embeddings

    if base in pivot:
        logger.warn(f"removing base '{base}' from comparison list '{pivot}'")
        pivot.remove(base)

    base = str(base)
    p = Path("output")

    out = []
    for dataset in datasets:
        for system in systems:
            for embedding in embeddings:
                if base_system is not None:
                    sys1 = p / base / dataset / embedding / "output.csv"
                else:
                    sys1 = p / system / dataset / base / "output.csv"
                sys2 = p / system / dataset / embedding / "output.csv"

                if base_system is not None:
                    system = base + "_" + system
                else:
                    embedding = base + "_" + embedding
                output = p / (f"{system}_{dataset}_{embedding}_pair_eval.csv")

                out.append((sys1, sys2, output))

    return out


# constants
dataset_lcstep = "lcstep"
dataset_recipenlg = "recipenlg"
dataset_champ = "champ"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="openai-gpt-3.5-turbo-0125",
        help="full name of service & model to use",
    )
    parser.add_argument("--gt", action="store_true", help="provide the gold steps in the prompt")
    parser.add_argument(
        "--nruns", type=int, default=3, help="number of times to run each of the two type"
    )
    parser.add_argument(
        "--workers", type=int, default=10, help="number of concurrent requests to make to the LLM"
    )

    args = parser.parse_args()

    logger = logging.getLogger("main")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    model = Model.from_full_name(args.model)

    datasets = ["champ", "lcstep", "recipenlg"]
    systems = [
        "zeroshot",
        "fewshot",
        "rag",
        "aag",
        "aag_no-critic",
        "aag_no-summ",
        "aag_no-summ_no-critic",
    ]
    embeddings = ["hf-nomic-embed-text-v1.5"]

    combos = build_combinations(logger, datasets, systems, embeddings, base_system="rag_no-critic")

    async def eval_all():
        for i, (sys1, sys2, output) in enumerate(combos):
            await pairwise_eval(logger, model, sys1, sys2, output, tqdm_desc=f"{i+1}/{len(combos)}")

    asyncio.run(eval_all())
