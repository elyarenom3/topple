#!/bin/bash
# E2 under perception with delta = 0.3 (towers grow), 24 towers, 3 low priority workers. Starts
# after the engine replay (run_engine_truth.sh) so that only one background job shares the
# machine with the renders.
set -u
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
while pgrep -f run_engine_truth.sh > /dev/null; do sleep 60; done
echo "start E2grow $(date)"
nice -n 19 python3 run_experiment.py --condition E2grow --profile study --towers 24 --workers 3 --quiet --delta 0.3 --out ../results
echo "E2grow done $(date)"
