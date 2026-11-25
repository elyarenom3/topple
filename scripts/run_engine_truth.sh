#!/bin/bash
# Replay E1all and E2all in MuJoCo and Drake at low priority (runs alongside the renders).
set -u
cd "$(dirname "$0")"
nice -n 19 python3 run_engine_truth.py --exp E1all --workers 3
nice -n 19 python3 run_engine_truth.py --exp E2all --workers 3
echo "engine truth done $(date)"
