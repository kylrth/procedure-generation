import asyncio
from typing import Any

from dataset import Procedure
from utils import log
import argparse
from systems import *
from pathlib import Path
import sys
import pandas as pd
from utils import log, spread_gather
from evaluation.api_overlap import *
from evaluation.edit_distance import *
from evaluation.is_all_inp_used import *
from evaluation.number_comparison import *
from evaluation.overall_score import *
from evaluation.rouge_score import *
from evaluation.tfidf import *
import json
from evaluation.heuristic import Heuristic

async def evaluate_all(
    sample_id: int,
    gold: Procedure,
    generated: list[str],
    evals: list[Heuristic]
):
    """Evaluate a generated procedure by comparing with the gold procedure using various methods.

    The returned dictionary contains the evaluation result for each metric.
    """
    results = {"_id": sample_id}
    async_tasks = {}
    
    for name, heuristic_obj in evals.items():
        try:
            results[name] = heuristic_obj.evaluate(gold, generated)
        except NotImplementedError:
            async_tasks[name] = heuristic_obj.aevaluate(gold, generated)
        
    print("Collecting the heuristics scores for the example")
    resp = await asyncio.gather(*async_tasks.values())
    print("Results of heuristics collected")
    # add async results to dict
    for name, result in zip(async_tasks.keys(), resp, strict=True):
        results[name] = result

    return results


def read_outputs_csv(args):
    output_dir = f'./output/{args.dataset}/output.csv'
    out_csv = pd.read_csv(output_dir,header=0,usecols=["question_id", "input", "output", "gold_steps", "completion"])
    out_list = []
    for i in range(len(out_csv)):
        row = out_csv.values[i]
        q_id = int(row[0])
        inputs = row[1]
        outputs= row[2]
        gold_steps = json.loads(row[3])
        gold = Procedure(_input=inputs, output=outputs, steps=gold_steps)
        gen_steps = json.loads(row[4])
        out_list.append((q_id, gold, gen_steps))
        
    return out_list


def int_leq(v: int):
    """Custom type conversion function which validates that the int is less than or equal to v."""

    def validate(value):
        i = int(value)
        if i > v:
            raise argparse.ArgumentTypeError(f"{value} is not less than {v}")

        return i

    return validate

# constants
dataset_lcstep = "lcstep"
dataset_recipenlg = "recipenlg"
dataset_champ = "champ"

def get_evaluations(dataset, model):
    evals = {}
    if dataset == dataset_recipenlg:
        evals["Ing_Used"] = All_Inp_Used(model, dataset)
        evals["ROUGE"] = ROUGE_Score()
        evals["TfIdf"] = TfIdf(dataset)
        evals["Edit-Distance"] = Edit_Distance(model)
        evals["Num-Compare"] = Number_Comparison(model)
        evals["Overall"] = Overall_Score(model)
    elif dataset == dataset_lcstep:
        evals["Api-Overlap"] = Api_Overlap()
        evals["Inp-Used"] = All_Inp_Used(model, dataset)
        evals["ROUGE"] = ROUGE_Score()
        evals["TfIdf"] = TfIdf(dataset)
        evals["Overall"] = Overall_Score(model)
    elif dataset == dataset_champ:
        evals["Overall"] = Overall_Score(model)
    
    return evals

async def get_results(out_list, dataset, model):
    # out_evals = []
    evals = get_evaluations(dataset, model)
    
    out_evals = await spread_gather(
            lambda sample_data: evaluate_all(*sample_data[1], evals),
            enumerate(out_list),
            5, #Num workers
            len(out_list),
        )
    return out_evals

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
        "-s",
        "--system",
        type=str,
        default="RAG",
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
        "-n", type=int, default=sys.maxsize, help="limit the number of samples to test"
    )
    parser.add_argument(
        "--workers", type=int, default=10, help="number of concurrent requests to make to the LLM"
    )
    parser.add_argument(
        "-k",
        type=int_leq(5),
        default=3,
        help=(
            "maximum number of examples to provide for FewShot and RAG (fewer are provided if they "
            "don't fit)"
        ),
    )

    args = parser.parse_args()
    cache_path = Path("cache")
    model = Model.from_full_name(args.model)
    print("Initialized model")
    
    #Read Outputs
    out_list = read_outputs_csv(args)
    print("Read the outputs csv file")
    out_evals = asyncio.run(get_results(out_list, args.dataset, model))

    df = pd.DataFrame(out_evals)
    df.to_csv(f"./output/{args.dataset}/eval_results.csv", header=True, index=False)