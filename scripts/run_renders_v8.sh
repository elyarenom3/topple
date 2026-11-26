#!/bin/bash
# H11 and V6 with the camera below the key softbox edge
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 render_stills.py H11 --width 1920 --spp 512
python3 render_videos.py V6 --width 960 --spp 16
echo "renders done $(date)"
