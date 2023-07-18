#!/bin/bash

set -e

# must run from the same directory
cd "$(dirname "$0")"

python scrape.py  # gets docs/concepts and docs/procedures/full
python format_procedure.py  # gets docs/procedures/formatted
python api_ref.py  # gets docs/api

# tar -cf langchain_procedures.tar.gz docs/
