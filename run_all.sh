#!/bin/bash
# call a command for all datasets, systems, and embedding models
# e.g. OPENAI_API_KEY=$(cat openai.key) ./run_all.sh python run.py

set -e

datasets=("champ" "lcstep" "recipenlg")
embedders=("hf-all-mpnet-base-v2")
systems=("zeroshot" "fewshot" "rag" "react" "aag")

for dataset in "${datasets[@]}"
do
    for embedder in "${embedders[@]}"
    do
        for system in "${systems[@]}"
        do
            flags="--dataset $dataset --embedder $embedder --system $system"
            echo "running with flags: $flags"
            $@ $flags

            # RAG has an optional critic
            if [ "$system" == "rag" ]
            then
                flags="--dataset $dataset --embedder $embedder --system $system --critic"
                echo "running with flags: $flags"
                $@ $flags
            fi

            # AAG has optional summarization and critic
            if [ "$system" == "aag" ]
            then
                flags="--dataset $dataset --embedder $embedder --system $system --summarize --critic"
                echo "running with flags: $flags"
                $@ $flags

                flags="--dataset $dataset --embedder $embedder --system $system --summarize"
                echo "running with flags: $flags"
                $@ $flags

                flags="--dataset $dataset --embedder $embedder --system $system --critic"
                echo "running with flags: $flags"
                $@ $flags
            fi
        done
    done
done
