#!/bin/bash
# call a command for all datasets, systems, and embedding models
# e.g. OPENAI_API_KEY=$(cat openai.key) ./run_all.sh python run.py

set -e

datasets=("champ" "lcstep" "recipenlg")
systems=("zeroshot" "fewshot" "rag" "react" "aag")

for dataset in "${datasets[@]}"
do
    for embedder in "${embedders[@]}"
    do
        for system in "${systems[@]}"
        do
            flags="--ds $dataset --system $system"
            echo "running with flags: $flags"
            $@ $flags

            # RAG has an optional critic and hierarchical retrieval
            if [ "$system" == "rag" ]
            then
                flags="--ds $dataset --system $system --critic"
                echo "running with flags: $flags"
                $@ $flags

                flags="--ds $dataset --system $system --hierarchical-retrieval"
                echo "running with flags: $flags"
                $@ $flags
            fi

            # AAG can ablate critic and hierarchical retrieval
            if [ "$system" == "aag" ]
            then
                flags="--ds $dataset --system $system --no-critic"
                echo "running with flags: $flags"
                $@ $flags

                flags="--ds $dataset --system $system --no-hierarchical-retrieval"
                echo "running with flags: $flags"
                $@ $flags

                flags="--ds $dataset --system $system --no-critic --no-hierarchical-retrieval"
                echo "running with flags: $flags"
                $@ $flags
            fi
        done
    done
done
