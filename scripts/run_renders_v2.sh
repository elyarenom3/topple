#!/bin/bash
# Renders after the second round of experiments: showcase tower (8 piles, keep the tallest),
# stills H1 to H12 at 1920 wide and 1024 samples per pixel, videos V1 to V9 at 960 wide, viewer.
set -u
cd "$(dirname "$0")"
while pgrep -f run_all_v2.sh > /dev/null; do sleep 60; done
echo "start renders $(date)"
./build_showcase_best.sh 8 24
echo "showcase done $(date)"
python3 render_stills.py all --width 1920 --spp 1024
echo "stills done $(date)"
python3 render_videos.py all --width 960 --spp 32
echo "videos done $(date)"
python3 export_viewer.py
echo "renders done $(date)"
