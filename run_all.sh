#!/bin/bash
# call a command for all datasets, systems, and embedding models
# e.g. ./run_all.sh OPENAI_API_KEY=$(cat openai.key) python run.py

set -e

datasets=("champ" "lcstep" "recipenlg -n 1000")
embedders=("hf-all-mpnet-base-v2" "openai-text-embedding-3-large")
systems=("RAG" "AAG")

for dataset in "${datasets[@]}"
do
    for embedder in "${embedders[@]}"
    do
        for system in "${systems[@]}"
        do
            flags="--dataset $dataset --embedder $embedder --system $system"
            echo "running with flags: $flags"
            $@ $flags
        done
    done
done
