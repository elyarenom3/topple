#!/bin/bash
# Follow up experiments at study scale, run after the E1all and E2all runs (Section 11.3).
# E4: risk weight sweep with delta 0.3 so towers can grow under uncertainty (fixed perturbation only).
# E5: cross design through the base point (noise scale and friction prior width).
# E6: one run per stone class. E3: Sobol indices. Arm demonstration.
set -u
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
W=${WORKERS:-10}
while pgrep -f "condition E1all|condition E2all" > /dev/null; do sleep 30; done
echo "start follow ups $(date)"
for k in 0 0.5 1 2; do
  python3 run_experiment.py --condition E4 --profile study --towers 10 --workers $W --quiet --kappa $k --delta 0.3 --probes 0 --tests fixed --out ../results
done
for ns in 0.5 2 4; do
  python3 run_experiment.py --condition E5 --profile study --towers 6 --workers $W --quiet --noise-scale $ns --probes 2 --tests fixed --out ../results
done
for fs in 0.05 0.1 0.5; do
  python3 run_experiment.py --condition E5 --profile study --towers 6 --workers $W --quiet --friction-log-sigma $fs --probes 2 --tests fixed --out ../results
done
python3 run_experiment.py --condition E5 --profile study --towers 6 --workers $W --quiet --probes 2 --tests fixed --out ../results
for cls in flat_river blocky_quarry rounded_cobble; do
  python3 run_experiment.py --condition E6 --profile study --towers 6 --workers $W --quiet --stone-class $cls --probes 2 --tests tilt,fixed --out ../results
done
OPENBLAS_NUM_THREADS=1 python3 fig_sobol.py --profile study --towers 2 --n 64 --max-stones 3 --classes flat_river,blocky_quarry,rounded_cobble
python3 arm_demo.py --profile study --towers 3 --max-stones 3
echo "follow ups done $(date)"
