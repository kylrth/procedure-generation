import argparse
import sys
import time
from math import ceil
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import recipenlg


def embed_batched(data: list[str], emb: SentenceTransformer, gpu: bool) -> np.ndarray:
    """Retries embedding with smaller and smaller batch size until it fits in memory."""
    batch = 32  # On CPU the batch size doesn't really matter, so we'll set to default.
    if gpu:
        batch = 256  # start high, will reduce until no OOM

    while True:
        pool = emb.start_multi_process_pool(None if gpu else ["cpu"] * 4)
        try:
            embeddings = emb.encode_multi_process(data, pool, batch_size=batch)
            break
        except RuntimeError as e:
            if "CUDA error: out of memory" not in str(e):
                raise
            batch //= 2
            if batch < 1:
                raise RuntimeError("insufficient GPU memory for even 1 example") from e
            print("CUDA OOM; reducing batch size to", batch, file=sys.stderr)
        finally:
            emb.stop_multi_process_pool(pool)


    return embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="generate HuggingFace embeddings and save them to disk"
    )
    parser.add_argument("subset", help="subset to create embeddings for (train/val/test)")
    parser.add_argument(
        "-d",
        "--data-dir",
        type=str,
        default="./data",
        help="directory containing the RecipeNLG dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="HuggingFace model to use for embeddings",
    )
    parser.add_argument(
        "--model-cache",
        type=str,
        default=None,
        help="model cache for sentence_transformers"
    )
    parser.add_argument(
        "--datasets-cache",
        type=str,
        default=None,
        help="cache dir for HF datasets"
    )
    parser.add_argument(
        "-o",
        "--output-root",
        type=str,
        default="embeddings",
        help="directory where embeddings will be stored",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=1,
        help=(
            "the job of embedding all the data is split into this many chunks, of which this "
            "process handles only the one specified by --chunk"
        ),
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=1,
        help="which chunk of the dataset this process will handle (one-indexed)",
    )
    parser.add_argument(
        "--no-gpu", action="store_true", help="don't use a GPU even if it's available"
    )

    args = parser.parse_args()

    # load data
    data = recipenlg.load(args.subset, args.data_dir, args.datasets_cache)

    start_time = time.time()

    # get the chunk
    chunk_length = int(ceil(len(data) / args.chunks))
    chunk_start = chunk_length * (args.chunk - 1)
    chunk_end = min(chunk_start + chunk_length, len(data))
    data = data.select(np.arange(chunk_start, chunk_end))

    # embed
    print(f"embedding {len(data)} samples", file=sys.stderr)
    emb = SentenceTransformer(args.model, cache_folder=args.model_cache)
    embeddings = embed_batched(
        data["formatted"], emb, gpu=torch.cuda.is_available() and not args.no_gpu
    )

    # save
    outfile: Path = Path(args.output_root) / args.model / f"{args.chunk}.txt.gz"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(outfile, embeddings)
    print(f"saved to {outfile}")

    elapsed = time.time() - start_time
    hours, minutes = divmod(elapsed, 3600)
    minutes, seconds = divmod(minutes, 60)
    print(f"finished in {int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}")
