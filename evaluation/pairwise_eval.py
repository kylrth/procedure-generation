import argparse
import asyncio
import json
import logging
import random
random.seed(1)
import re
import sys
from pathlib import Path
import os

import pandas as pd
import numpy as np
from langchain.schema import HumanMessage, SystemMessage

from dataset import Procedure, format_steps
from systems import Model
from utils import spread_gather


def prepare_prompt(proc1_steps: str, proc2_steps: str, gold: Procedure, use_gt: bool):
    if not use_gt:
        prompt = [
            SystemMessage(
                content=(
                    "[Instruction]\n"
                    "Please act as an impartial judge and evaluate the quality of the two answer "
                    "procedures provided below to achieve the same user goal and choose one out of the "
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
    else:
        prompt = [
            SystemMessage(
                content=(
                    "[Instruction]\n"
                    "Please act as an impartial judge and evaluate the quality of the two answer "
                    "procedures provided below to achieve the same user goal and choose one out of the "
                    "two as the preferred procedure. For your reference, a gold procedure is given below"
                    " which accurately achieves the user goal.\n"
                    "For this evaluation, you should primarily evaluate if the procedure achieves the "
                    "user goal adequately or not and whether it uses all the resources mentioned in "
                    "user goal or not. Do not penalize a procedure based on its sentences' structure, "
                    "grammar and wording, but focus on the facts and suggestions being made by the "
                    "procedure; if the procedure resembles closely with the gold procedure and leads "
                    "to the specified user goal with sufficient "
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
                    f"[BEGIN GOLD]\n{gold.format_steps()}\n[END GOLD]\n\n"
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
    match = re.findall(r"\s*\[\[(\d+)\]\]\s*$", sentence) #Check pattern at end of string [[<score>]]
    if not match:
        match = re.findall(r"\s*\[\[\w*:\s*(\d+)\]\]\s*$", sentence) #Check at end of string [[Choice: <score>]]
        if not match:
            match = re.findall(r"\s*\[\[(\d+)\]\]\s*", sentence) #Check anywhere in string [[<score>]]
            if not match:
                match = re.findall(r"\s*\[\[\w*:\s*(\d+)\]\]\s*", sentence) #Check anywhere in string [[Choice: <score>]]
                if not match:
                    match = re.search(r"\s*(N\/A)\s*$", sentence) #Check any form of N/A anywhere
                    if not match:
                        return -3
                    return -1

    return int(match[-1])

async def get_choice(model:Model, prompt, seed: int, if_inv: bool) -> int:
    answer = await model.generate(prompt, seed=seed)
    score = result_parser(answer)
    if if_inv:
        if score == 1:
            score = 2
        elif score == 2:
            score = 1    
    return score

def get_final_choice(score: list[int])->int:
    score = np.array(score)
    score_rag = np.sum(score==1)
    score_aag = np.sum(score==2)
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
    seed_list = [random.randint(0,sys.maxsize) for _ in range(2*num_eval_each_type)]
    prompt = prepare_prompt(format_steps(sys1_generated), format_steps(sys2_generated), gold, use_gt)
    prompt_inv = prepare_prompt(format_steps(sys2_generated), format_steps(sys1_generated), gold, use_gt)
    prompt_list = [pt for _ in range(num_eval_each_type) for pt in [prompt, prompt_inv]]
    logger.debug("Prompt prepared")
    score = await spread_gather(
        lambda pt: get_choice(model, pt[1][0], pt[1][1], if_inv=(pt[0]%2==1)),
        enumerate(zip(prompt_list, seed_list)),
        5,
        len(prompt_list)
    )
    final_choice = get_final_choice(score)
    logger.debug(f"LLM choice was {final_choice}")
    res_dict = {"question_id": q_id, "choice": final_choice}
    return res_dict


async def get_results(logger, sys1_out_list, sys2_out_list, model, use_gt, num_eval_each_type):
    out_evals = await spread_gather(
        lambda sample_data: aevaluate(logger, model, use_gt, num_eval_each_type, *sample_data[1][0], sample_data[1][1][2]),
        enumerate(zip(sys1_out_list, sys2_out_list)),
        5,  # Num workers
        len(sys1_out_list),
    )
    return out_evals


def read_outputs_csv(args, system_name, embedder=None):
    if embedder is not None:
        output_dir = f"./output/{system_name}/{args.dataset}/{embedder}/output.csv"
    else:
        output_dir = f"./output/{system_name}/{args.dataset}/{args.embedder}/output.csv"
    out_csv = pd.read_csv(
        output_dir, header=0, usecols=["question_id", "input", "output", "gold_steps", "completion"]
    )
    out_csv = out_csv.sort_values(by=["question_id"])

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
        default="openai-gpt-3.5-turbo-0125",
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
        "--gt", action="store_true", help="limit the number of samples to test"
    )
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
    logger.setLevel(logging.DEBUG)
    logger.debug(f"Running for: {args}")

    cache_path = Path("cache")
    model = Model.from_full_name(args.model)
    logger.info("Initialized model")
    args.system1 = args.system1.lower()
    args.system2 = args.system2.lower()

    sys1_out_list = read_outputs_csv(args, args.system1, "hf-all-mpnet-base-v2")
    sys2_out_list = read_outputs_csv(args, args.system2, "openai-text-embedding-3-large")

    out_evals = asyncio.run(get_results(logger, sys1_out_list, sys2_out_list, model, args.gt, args.nruns))

    to_record = pd.DataFrame(out_evals)
    if args.gt:
        prefix = 'with-gt'
    else:
        prefix = 'without-gt-between-embeds'
    
    os.makedirs(f'./output/{prefix}', exist_ok=True)
    to_record.to_csv(
        f"./output/{prefix}/{args.system1}_{args.system2}_{args.dataset}_{args.embedder}_pair_eval.csv",
        header=True,
        index=False,
    )