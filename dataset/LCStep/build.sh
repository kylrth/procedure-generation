#!/bin/bash

set -e

# allow imports from root of repo
export PYTHONPATH=$PYTHONPATH:$(dirname $(pwd))

# must run from the same directory
cd "$(dirname "$0")"

python scrape.py  # gets docs/concepts and docs/procedures/full
python api_ref.py  # gets docs/api
OPENAI_API_KEY=$(cat ../../openai.key) python format_procedure.py  # gets docs/procedures/formatted
OPENAI_API_KEY=$(cat ../../openai.key) python procedure_checks.py  # checks formatted procedures

# tar -czf langchain_procedures.tar.gz docs/
