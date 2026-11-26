# Predictions scored against MuJoCo and Drake

Written by `scripts/fig_engine_truth.py` from the replays of `scripts/run_engine_truth.py`. Intervals: 95% bootstrap over towers.

## E1all (246 placements, 24 towers)

Replay check in the topple's engine: 64 of 67 building placements give the recorded outcome; largest critical tilt difference 3.40 degrees.

| Engine | Survival rate | Same outcome as topple's engine | Critical tilt, mean difference (deg) | Spearman |
| --- | --- | --- | --- | --- |
| topple | 0.34 |  |  |  |
| mujoco | 0.25 | 0.84 | 2.37 | 0.79 |
| drake | 0.23 | 0.87 | 1.93 | 0.83 |

| Prediction | Truth from | Brier | ECE | AUC | Margin bias (deg) | Margin RMSE (deg) | Margin Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tilt margin | topple | 0.329 [0.282, 0.371] | 0.329 [0.279, 0.374] | 0.73 | +2.88 | 5.62 | 0.75 |
| tilt margin | mujoco | 0.427 [0.376, 0.471] | 0.427 [0.377, 0.473] | 0.68 | +4.12 | 6.97 | 0.62 |
| tilt margin | drake | 0.427 [0.380, 0.464] | 0.427 [0.384, 0.465] | 0.69 | +4.57 | 6.97 | 0.64 |
| rocking margin | topple | 0.195 [0.153, 0.241] | 0.195 [0.149, 0.239] | 0.81 | -0.33 | 3.18 | 0.75 |
| rocking margin | mujoco | 0.260 [0.198, 0.317] | 0.260 [0.200, 0.321] | 0.77 | +0.91 | 4.20 | 0.61 |
| rocking margin | drake | 0.236 [0.179, 0.295] | 0.236 [0.177, 0.295] | 0.81 | +1.36 | 3.81 | 0.66 |

## E2all (143 placements, 24 towers)

Replay check in the topple's engine: 25 of 25 building placements give the recorded outcome; largest critical tilt difference 0.32 degrees.

| Engine | Survival rate | Same outcome as topple's engine | Critical tilt, mean difference (deg) | Spearman |
| --- | --- | --- | --- | --- |
| topple | 0.46 |  |  |  |
| mujoco | 0.41 | 0.85 | 2.16 | 0.84 |
| drake | 0.38 | 0.90 | 1.77 | 0.86 |

| Prediction | Truth from | Brier | ECE | AUC | Margin bias (deg) | Margin RMSE (deg) | Margin Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Monte Carlo | topple | 0.209 [0.171, 0.254] | 0.156 [0.117, 0.230] | 0.75 | -2.34 | 6.15 | 0.61 |
| Monte Carlo | mujoco | 0.166 [0.132, 0.205] | 0.121 [0.101, 0.208] | 0.82 | -1.10 | 4.67 | 0.63 |
| Monte Carlo | drake | 0.171 [0.138, 0.208] | 0.113 [0.084, 0.205] | 0.82 | -0.91 | 4.50 | 0.70 |
| point estimate | topple | 0.441 [0.375, 0.507] | 0.441 [0.370, 0.504] | 0.58 | +1.65 | 7.79 | 0.55 |
| point estimate | mujoco | 0.441 [0.376, 0.514] | 0.441 [0.369, 0.514] | 0.61 | +2.89 | 7.60 | 0.56 |
| point estimate | drake | 0.462 [0.389, 0.536] | 0.462 [0.383, 0.532] | 0.60 | +3.08 | 7.58 | 0.57 |
| unscented | topple | 0.245 [0.210, 0.282] | 0.225 [0.171, 0.290] | 0.75 | -0.71 | 6.36 | 0.58 |
| unscented | mujoco | 0.233 [0.200, 0.271] | 0.264 [0.202, 0.333] | 0.82 | +0.52 | 5.33 | 0.59 |
| unscented | drake | 0.238 [0.205, 0.273] | 0.274 [0.219, 0.337] | 0.82 | +0.71 | 5.29 | 0.64 |
| rocking margin, Monte Carlo | topple | 0.222 [0.172, 0.273] | 0.182 [0.148, 0.255] | 0.74 | -3.97 | 7.77 | 0.56 |
| rocking margin, Monte Carlo | mujoco | 0.164 [0.119, 0.211] | 0.132 [0.106, 0.209] | 0.82 | -2.73 | 5.95 | 0.60 |
| rocking margin, Monte Carlo | drake | 0.170 [0.132, 0.217] | 0.120 [0.094, 0.198] | 0.80 | -2.54 | 5.92 | 0.65 |
| Monte Carlo with learned recalibration | topple | 0.168 [0.135, 0.201] | 0.069 [0.064, 0.171] | 0.83 |  | |  |
| Monte Carlo with learned recalibration | mujoco | 0.130 [0.098, 0.167] | 0.074 [0.062, 0.148] | 0.89 |  | |  |
| Monte Carlo with learned recalibration | drake | 0.158 [0.125, 0.189] | 0.087 [0.071, 0.174] | 0.84 |  | |  |
