#!/bin/bash
# Step 4: the frozen recalibrator on fresh stone sets (towers 24 to 47)
cd "$(dirname "$0")"
LOG=../results/logs/e2fresh.log
mkdir -p ../results/logs
echo "start E2fresh $(date)" >> $LOG
OMP_NUM_THREADS=1 nice -n 19 python3 run_experiment.py --condition E2fresh --profile study --towers 24 --tower-start 24 --workers 3 --quiet --delta 0.5 --out ../results >> $LOG 2>&1
echo "E2fresh finished $(date)" >> $LOG
