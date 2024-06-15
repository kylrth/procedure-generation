#!/bin/bash
# call a command for all datasets, systems, and embedding models
# e.g. OPENAI_API_KEY=$(cat openai.key) ./run_all.sh python run.py

set -e

datasets=("champ" "lcstep" "recipenlg")
embedders=("hf-nomic-embed-text-v1.5")
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