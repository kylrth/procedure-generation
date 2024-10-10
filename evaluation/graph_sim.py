import argparse
import asyncio
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

from dataset import GraphProcedure
from embedder import CachingEmbedder, Embedder, embedder_from_name
from utils import spread_gather


dataset_lcstep = "lcstep"
dataset_recipenlg = "recipenlg"
dataset_champ = "champ"


async def calc_similarity(id_: int, file_path, embedder: Embedder):
    gen_graph: GraphProcedure = None
    gt_graph: GraphProcedure = None
    with file_path.open("rb") as f:
        gt_graph, gen_graph = pickle.load(f)

    gen_embed = await gen_graph.get_graph_embedding(embedder)
    gt_embed = await gt_graph.get_graph_embedding(embedder)
    cos_sim = np.dot(gen_embed, gt_embed) / (np.linalg.norm(gen_embed) * np.linalg.norm(gt_embed))
    return_dict = {"id": id_, "sim": round(cos_sim, 3)}
    return return_dict


def evaluate_and_store(args):
    system_name = args.system.lower()
    out_name = (
        system_name
        # + ("_hs" if system_name in ("aag", "rag", "react") and args.hs else "_s")
        + ("_no-critic" if system_name in ("rag", "aag") and not args.critic else "")
        + (f"_{args.n_queries}" if system_name == "aag" and args.n_queries != 4 else "")
    )
    outdir = Path("output") / out_name / args.dataset / args.embedder
    cache_path = Path("cache") / system_name / args.dataset / args.embedder
    embedder = CachingEmbedder(embedder_from_name(args.embedder), cache_path)
    file_list = list(outdir.glob("*.pkl"))
    breakpoint()
    sim_results = asyncio.run(
        spread_gather(
            lambda item: calc_similarity(*item, embedder),
            enumerate(file_list),
            min(args.workers, args.n),
            len(file_list),
        )
    )

    sim_vals = []
    for res in sim_results:
        sim_vals.append(res["sim"])

    print(f"DATASET: {args.dataset}, SYSTEM: {system_name}, NUM_EXAMPLES: {len(file_list)}")
    print(f"Mean: {np.mean(sim_vals)}, Std: {np.std(sim_vals)}")
    csv_path = outdir / "similarity.csv"
    with csv_path.open("w", newline="") as out_file:
        dict_writer = csv.DictWriter(out_file, sim_results[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(sim_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "-em",
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
    parser.add_argument(
        "--n-queries",
        type=int,
        default=4,
        help="Number of re-written queries",
    )
    parser.add_argument(
        "--critic", action="store_true", help="Whether to use critic in the system or not"
    )
    parser.add_argument(
        "--hs", action="store_true", help="Whether to use hierarchical search in the system or not"
    )
    args = parser.parse_args()
    evaluate_and_store(args)
