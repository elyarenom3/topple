#!/bin/bash
# Second round of study scale experiments, after the contact fixes (touching hull query,
# load bearing manifold points, settle wobble) and with the rocking aware margin (Margin 4).
# Order: E1all and E2all, E4 with arm placement, E6, E5, Sobol (E3), arm demonstration, F4.
set -u
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
W=${WORKERS:-11}
T=${TOWERS:-24}
R=../results
echo "start $(date)"
python3 run_experiment.py --condition E1all --profile study --towers $T --workers $W --quiet --out $R
echo "E1all done $(date)"
python3 run_experiment.py --condition E2all --profile study --towers $T --workers $W --quiet --out $R
echo "E2all done $(date)"
for k in 0 0.5 1 2; do
  python3 run_experiment.py --condition E4 --profile study --towers $T --workers $W --quiet --kappa $k --delta 0.3 --probes 0 --tests fixed --placement arm --tag arm --out $R
  echo "E4 kappa $k done $(date)"
done
for cls in flat_river blocky_quarry rounded_cobble; do
  python3 run_experiment.py --condition E6 --profile study --towers $T --workers $W --quiet --stone-class $cls --probes 2 --tests tilt,fixed --out $R
  echo "E6 $cls done $(date)"
done
python3 run_experiment.py --condition E5 --profile study --towers $T --workers $W --quiet --probes 2 --tests fixed --out $R
for ns in 0.5 2 4; do
  python3 run_experiment.py --condition E5 --profile study --towers $T --workers $W --quiet --noise-scale $ns --probes 2 --tests fixed --out $R
done
for fs in 0.05 0.1 0.5; do
  python3 run_experiment.py --condition E5 --profile study --towers $T --workers $W --quiet --friction-log-sigma $fs --probes 2 --tests fixed --out $R
done
echo "E5 done $(date)"
python3 fig_sobol.py --profile study --towers 4 --n 256 --max-stones 3 --classes flat_river,blocky_quarry,rounded_cobble
echo "Sobol done $(date)"
python3 arm_demo.py --profile study --towers 3 --max-stones 3
echo "arm demo done $(date)"
python3 fig_engine_comparison.py --profile study --towers 12
../.conda311/bin/python fig_engine_comparison.py --profile study --towers 12 --engines pybullet --merge
echo "all done $(date)"
