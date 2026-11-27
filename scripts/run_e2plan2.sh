#!/bin/bash
# Step 3: E2plan2 (sequential screen, acceptance on a fresh full evaluation, 48 candidates)
cd "$(dirname "$0")"
LOG=../results/logs/e2plan2.log
mkdir -p ../results/logs
echo "start E2plan2 $(date)" >> $LOG
OMP_NUM_THREADS=1 nice -n 19 python3 run_experiment.py --condition E2plan2 --profile study --towers 24 --workers 3 --quiet --delta 0.5 --out ../results >> $LOG 2>&1
echo "E2plan2 finished $(date)" >> $LOG
