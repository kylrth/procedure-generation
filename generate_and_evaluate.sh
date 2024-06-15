#!/bin/bash

set -e

./run_all.sh python run.py
./run_all.sh python evaluation/eval.py
python evaluation/pairwise_eval.py --nruns 5 --workers 20
