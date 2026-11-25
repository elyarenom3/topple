#!/bin/bash
# Unscented transform predictions on the same perceived placements as E2all (towers 0 to 11):
# the placements are reproduced exactly because every random stream is keyed by tower and stage.
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
while pgrep -f run_followups.sh > /dev/null; do sleep 30; done
python3 run_experiment.py --condition E2ut --profile study --towers 12 --workers 10 --quiet --tests tilt,fixed --out ../results
