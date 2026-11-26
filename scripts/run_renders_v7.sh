#!/bin/bash
# V6 after the camera fix, once the H11 and H12 stills are done
cd "$(dirname "$0")"
until grep -q "stills done" ../results/logs/run_renders_v6.log 2>/dev/null; do sleep 30; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 render_videos.py V6 --width 960 --spp 16
echo "video done $(date)"
