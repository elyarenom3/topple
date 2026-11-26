#!/bin/bash
# Remaining stills after the H7 empty section fix
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 render_stills.py H7,H8,H9,H10,H11,H12 --width 1920 --spp 512
echo "stills done $(date)"
