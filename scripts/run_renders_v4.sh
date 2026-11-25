#!/bin/bash
# The stills after H1 at 1920 wide and 512 samples per pixel. The first batch in
# run_renders_v3.sh stopped at H2 (the subject was closer than the traced lens can focus); this
# starts when run_renders_v3.sh (H5, videos, viewer) has finished.
set -u
cd "$(dirname "$0")"
while pgrep -f run_renders_v3.sh > /dev/null; do sleep 60; done
echo "start stills v4 $(date)"
python3 render_stills.py H2,H3,H4,H6,H7,H8,H9,H10,H11,H12 --width 1920 --spp 512
echo "stills done $(date)"
