#!/bin/bash
# This script runs automatic checks of the compiled dataset.

base_path="docs/procedures"

# list of files in full/ directory
full_files=($(find "$base_path/full" -type f))

CODE=0

for file_path in "${full_files[@]}"; do

  # get formatted path
  formatted_path="${file_path/full/formatted}"

  # check if the formatted file exists
  if [ ! -f "$formatted_path" ]; then
    if [ "$formatted_path" == "docs/procedures/formatted/docs/use_cases/agent_simulations/multiagent_authoritarian.md" ]; then
      continue
    fi
    echo "missing $formatted_path"
    CODE=1
  fi
done

exit ${CODE}
