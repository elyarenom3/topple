#!/bin/bash
# Diagnostic runs for tower height under perception (E2plan settings, 24 towers, delta 0.5):
# E2cand (48 candidate poses per stone) and E2view (8 in hand views at two elevations, 640 x 480).
cd "$(dirname "$0")"
LOG=../results/logs/diagnostics.log
for c in E2cand E2view; do
  echo "start $c $(date)" >> $LOG
  OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 nice -n 19 python3 run_experiment.py --condition $c --profile study --towers 24 --workers 3 --quiet --delta 0.5 --out ../results >> $LOG 2>&1
  echo "$c finished $(date)" >> $LOG
done
echo "diagnostics complete $(date)" >> $LOG
