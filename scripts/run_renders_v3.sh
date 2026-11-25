#!/bin/bash
# Renders with reduced sample counts: stills 1920 wide at 512 samples per pixel (H5, the posterior
# ghosts, at 128), videos 960 x 540 at 16 samples per pixel, then the viewer. Uses the showcase
# tower built by run_renders_v2.sh.
set -u
cd "$(dirname "$0")"
echo "start renders v3 $(date)"
python3 render_stills.py H1,H2,H3,H4,H6,H7,H8,H9,H10,H11,H12 --width 1920 --spp 512
echo "stills done $(date)"
python3 render_stills.py H5 --width 1920 --spp 128
echo "H5 done $(date)"
python3 render_videos.py all --width 960 --spp 16
echo "videos done $(date)"
python3 export_viewer.py
echo "renders done $(date)"
