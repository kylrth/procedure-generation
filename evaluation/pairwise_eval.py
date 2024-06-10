import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd
from langchain.schema import HumanMessage, SystemMessage

from dataset import Procedure, format_steps
from systems import Model
from utils import spread_gather


def prepare_prompt(proc1_steps: str, proc2_steps: str, gold: Procedure):
    prompt = [
        SystemMessage(
            content=(
                "[Instruction]\n"
                "Please act as an impartial judge and evaluate the quality of the two answer "
                "proceduresprovided below to achieve the same user goal and choose one out of the "
                "two as the preferred procedure.\n"
                "For this evaluation, you should primarily evaluate if the procedure achieves the "
                "user goal adequately or not and whether it uses all the resources mentioned in "
                "user goal or not. Do not penalize a procedure based on its sentences' structure, "
                "grammar and wording, but focus on the facts and suggestions being made by the "
                "procedure; if the procedure leads to the specified user goal with sufficient "
                "details and is easy to understand on each intermediate step, then that procedure "
                "should be chosen.\n\n"
                "Compare the two procedures below and then provide a short explanation of your "
                "reason for the choice between the two. Be as objective as possible. Based on your "
                'explanation, you must choose one procedure following this format: "[[choice]]" '
                'where choice can be 1 or 2 only, for example: "Choice: [[2]]". Print this choice '
                "at the END only.\n\n"
            )
        ),
        HumanMessage(
            content=(
                "[User Goal]\n"
                f"{gold.output} using {gold.input_}\n\n"
                f"[BEGIN PROCEDURE 1]\n{proc1_steps}\n[END OF PROCEDURE 1]\n\n"
                f"[BEGIN PROCEDURE 2]\n{proc2_steps}\n[END OF PROCEDURE 2]\n\n"
                f'Your answer should begin with "Here is my analysis of the comparison between the '
                "two procedures:\n1."
            )
        ),
    ]

    return prompt


def result_parser(sentence: str) -> int:
    if not sentence:
        return -2
    match = re.findall(r"\s*\[\[(\d+)\]\]\s*$", sentence)
    if not match:
        match = re.findall(r"\s*\[\[(\d+)\]\]\s*", sentence)
        if not match:
            match = re.search(r"\s*(N\/A)\s*$", sentence)
            if not match:
                return -3
            return 1

    return int(match[-1])


async def aevaluate(
    logger: logging.Logger,
    model: Model,
    q_id: int,
    gold: Procedure,
    sys1_generated: list[str],
    sys2_generated: list[str],
) -> dict[str, int]:
    prompt = prepare_prompt(format_steps(sys1_generated), format_steps(sys2_generated), gold)
    logger.debug("Prompt prepared")
    answer = await model.generate(prompt)
    if answer.find("[") == -1:
        logger.debug("Generating the response again")
        answer = await model.generate(prompt)
    logger.debug("Got model response for the choice")
    score = result_parser(answer)
    logger.debug("Returning the LLM choice!")
    res_dict = {"question_id": q_id, "choice": score}
    return res_dict


async def get_results(logger, sys1_out_list, sys2_out_list, model):
    out_evals = await spread_gather(
        lambda sample_data: aevaluate(logger, model, *sample_data[1][0], sample_data[1][1][2]),
        enumerate(zip(sys1_out_list, sys2_out_list)),
        5,  # Num workers
        len(sys1_out_list),
    )
    return out_evals


def read_outputs_csv(args, system_name):
    output_dir = f"./output/{system_name}/{args.dataset}/{args.embedder}/output.csv"
    out_csv = pd.read_csv(
        output_dir, header=0, usecols=["question_id", "input", "output", "gold_steps", "completion"]
    )
    out_list = []
    for i in range(len(out_csv)):
        row = out_csv.to_numpy()[i]
        q_id = int(row[0])
        inputs = row[1]
        outputs = row[2]
        gold_steps = json.loads(row[3])
        gold = Procedure(input_=inputs, output=outputs, steps=gold_steps)
        gen_steps = json.loads(row[4])
        out_list.append((q_id, gold, gen_steps))

    return out_list


# constants
dataset_lcstep = "lcstep"
dataset_recipenlg = "recipenlg"
dataset_champ = "champ"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data-dir",
        type=str,
        default="./dataset",
        help="path to the dataset dir",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=dataset_lcstep,
        choices=[dataset_lcstep, dataset_recipenlg, dataset_champ],
        help="Dataset to run the system on",
    )

    parser.add_argument(
        "-s1",
        "--system1",
        type=str,
        default="RAG",
        help="system to perform generation",
    )
    parser.add_argument(
        "-s2",
        "--system2",
        type=str,
        default="AAG",
        help="system to perform generation",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="openai-gpt-3.5-turbo-0613",
        help="full name of service & model to use",
    )
    parser.add_argument(
        "-e",
        "--embedder",
        type=str,
        default="hf-all-mpnet-base-v2",
        help="full name of service & model to use for embeddings",
    )
    parser.add_argument(
        "-n", type=int, default=sys.maxsize, help="limit the number of samples to test"
    )
    parser.add_argument(
        "--workers", type=int, default=10, help="number of concurrent requests to make to the LLM"
    )

    args = parser.parse_args()

    logger = logging.getLogger("main")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.debug(f"Running for: {args}")

    cache_path = Path("cache")
    model = Model.from_full_name(args.model)
    logger.info("Initialized model")
    args.system1 = args.system1.lower()
    args.system2 = args.system2.lower()

    sys1_out_list = read_outputs_csv(args, args.system1)
    sys2_out_list = read_outputs_csv(args, args.system2)

    out_evals = asyncio.run(get_results(logger, sys1_out_list, sys2_out_list, model))

    to_record = pd.DataFrame(out_evals)
    to_record.to_csv(
        f"./output/{args.system1}_{args.system2}_{args.dataset}_{args.embedder}_pair_eval.csv",
        header=True,
        index=False,
    )
