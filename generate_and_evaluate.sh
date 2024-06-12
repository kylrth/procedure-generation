#!/bin/bash

set -e

./run_all.sh python run.py
./run_all.sh python evaluation/eval.py

# pairwise evaluation
datasets=("champ" "lcstep" "recipenlg -n 100")
embedders=("hf-all-mpnet-base-v2" "openai-text-embedding-3-large")
for dataset in "${datasets[@]}"
do
    for embedder in "${embedders[@]}"
    do
        python evaluation/pairwise_eval.py --system1 AAG --system2 AAG \
            --dataset $dataset --embedder $embedder --nruns 5
    done
done
