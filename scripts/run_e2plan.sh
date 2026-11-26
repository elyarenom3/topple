#!/bin/bash
# E2 planning with the learned recalibration in the chance constraint at delta 0.5, after the renders
cd "$(dirname "$0")"
until grep -q "renders done" ../results/logs/run_renders_v8.log 2>/dev/null; do sleep 60; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
echo "start E2plan $(date)"
nice -n 19 python3 run_experiment.py --condition E2plan --profile study --towers 24 --workers 3 --quiet --delta 0.5 --out ../results
echo "E2plan done $(date)"
