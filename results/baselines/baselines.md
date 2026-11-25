# Baselines for the survival probability

Written by `scripts/fig_baselines.py` from the saved experiment records; leave one tower out validation.

### E2all: estimated shapes, every method on the same placements

143 placements, 24 towers. Intervals: 95%, bootstrap over towers. Differences are paired, relative to Monte Carlo (main method).

| Method | Group | ECE | Brier | AUC | ECE difference | Brier difference |
| --- | --- | --- | --- | --- | --- | --- |
| Monte Carlo (main method) | stated by the system | 0.156 [0.119, 0.229] | 0.209 [0.169, 0.253] | 0.754 [0.680, 0.823] |  |  |
| point estimate | stated by the system | 0.441 [0.372, 0.503] | 0.441 [0.372, 0.503] | 0.579 [0.524, 0.631] | +0.284 [+0.195, +0.340] | +0.231 [+0.147, +0.311] |
| unscented | stated by the system | 0.225 [0.173, 0.295] | 0.245 [0.209, 0.282] | 0.752 [0.673, 0.830] | +0.068 [-0.003, +0.126] | +0.036 [-0.001, +0.069] |
| first order | stated by the system | 0.219 [0.161, 0.296] | 0.257 [0.221, 0.294] | 0.711 [0.629, 0.787] | +0.062 [-0.019, +0.135] | +0.048 [+0.008, +0.088] |
| rocking margin | stated by the system | 0.182 [0.150, 0.255] | 0.222 [0.174, 0.270] | 0.741 [0.678, 0.811] | +0.025 [-0.025, +0.086] | +0.013 [-0.013, +0.039] |
| logistic on point estimate margin | curve fit | 0.078 [0.060, 0.156] | 0.204 [0.178, 0.232] | 0.716 [0.632, 0.798] | -0.078 [-0.130, +0.002] | -0.005 [-0.035, +0.022] |
| isotonic on point estimate margin | curve fit | 0.084 [0.060, 0.175] | 0.203 [0.166, 0.244] | 0.683 [0.582, 0.778] | -0.073 [-0.138, +0.017] | -0.007 [-0.038, +0.023] |
| logistic on Monte Carlo mean margin | curve fit | 0.084 [0.056, 0.168] | 0.190 [0.163, 0.220] | 0.734 [0.658, 0.807] | -0.072 [-0.120, -0.009] | -0.019 [-0.038, +0.000] |
| Monte Carlo, isotonic recalibration | curve fit | 0.103 [0.077, 0.198] | 0.198 [0.164, 0.237] | 0.701 [0.626, 0.780] | -0.054 [-0.099, +0.026] | -0.011 [-0.030, +0.008] |
| boosted trees, point estimate features | learned (E2all) | 0.098 [0.070, 0.181] | 0.189 [0.145, 0.231] | 0.796 [0.719, 0.871] | -0.059 [-0.121, +0.019] | -0.020 [-0.066, +0.022] |
| logistic, point estimate features | learned (E2all) | 0.094 [0.072, 0.187] | 0.186 [0.155, 0.224] | 0.793 [0.711, 0.857] | -0.062 [-0.110, +0.019] | -0.023 [-0.057, +0.009] |
| boosted trees, with Monte Carlo features | learned (E2all) | 0.126 [0.089, 0.210] | 0.199 [0.149, 0.249] | 0.786 [0.696, 0.866] | -0.030 [-0.085, +0.037] | -0.010 [-0.046, +0.027] |
| logistic, with Monte Carlo features | learned (E2all) | 0.143 [0.101, 0.234] | 0.197 [0.160, 0.235] | 0.780 [0.702, 0.851] | -0.014 [-0.073, +0.059] | -0.012 [-0.042, +0.016] |
| boosted trees, all experiments | learned (1757 placements) | 0.074 [0.066, 0.171] | 0.170 [0.140, 0.201] | 0.829 [0.771, 0.879] | -0.082 [-0.130, +0.018] | -0.039 [-0.077, -0.003] |
| logistic, all experiments | learned (1757 placements) | 0.084 [0.060, 0.182] | 0.172 [0.144, 0.205] | 0.819 [0.757, 0.873] | -0.073 [-0.127, +0.025] | -0.037 [-0.067, -0.006] |

### E1all: exact shapes, curve fits per margin type

246 placements, 24 towers. Intervals: 95%, bootstrap over towers. Differences are paired, relative to logistic on tilt margin.

| Method | Group | ECE | Brier | AUC | ECE difference | Brier difference |
| --- | --- | --- | --- | --- | --- | --- |
| logistic on tilt margin | curve fit | 0.065 [0.049, 0.120] | 0.105 [0.078, 0.135] | 0.897 [0.851, 0.946] |  |  |
| isotonic on tilt margin | curve fit | 0.063 [0.042, 0.116] | 0.111 [0.082, 0.140] | 0.860 [0.805, 0.919] | -0.002 [-0.039, +0.036] | +0.006 [-0.001, +0.012] |
| logistic on support polygon margin | curve fit | 0.055 [0.048, 0.113] | 0.139 [0.115, 0.165] | 0.846 [0.799, 0.895] | -0.010 [-0.046, +0.038] | +0.034 [+0.018, +0.048] |
| isotonic on support polygon margin | curve fit | 0.093 [0.074, 0.138] | 0.140 [0.115, 0.166] | 0.820 [0.774, 0.871] | +0.028 [-0.017, +0.064] | +0.035 [+0.019, +0.051] |
| logistic on push margin | curve fit | 0.057 [0.048, 0.122] | 0.116 [0.091, 0.146] | 0.880 [0.834, 0.930] | -0.008 [-0.025, +0.031] | +0.012 [+0.004, +0.021] |
| isotonic on push margin | curve fit | 0.073 [0.053, 0.114] | 0.122 [0.096, 0.151] | 0.847 [0.794, 0.905] | +0.008 [-0.036, +0.040] | +0.017 [+0.008, +0.028] |
| logistic on force closure margin | curve fit | 0.042 [0.026, 0.103] | 0.185 [0.160, 0.214] | 0.702 [0.635, 0.773] | -0.023 [-0.070, +0.032] | +0.080 [+0.056, +0.104] |
| isotonic on force closure margin | curve fit | 0.126 [0.091, 0.190] | 0.182 [0.158, 0.211] | 0.688 [0.629, 0.750] | +0.061 [-0.000, +0.107] | +0.077 [+0.053, +0.104] |
| logistic on rocking margin | curve fit | 0.067 [0.043, 0.112] | 0.106 [0.082, 0.131] | 0.864 [0.816, 0.913] | +0.002 [-0.049, +0.041] | +0.001 [-0.014, +0.019] |
| isotonic on rocking margin | curve fit | 0.045 [0.034, 0.091] | 0.103 [0.075, 0.131] | 0.814 [0.762, 0.873] | -0.020 [-0.059, +0.015] | -0.002 [-0.022, +0.019] |
| tilt margin as stated (exact shapes) | stated by the system | 0.329 [0.280, 0.374] | 0.329 [0.280, 0.374] | 0.734 [0.699, 0.771] | +0.265 [+0.179, +0.304] | +0.225 [+0.159, +0.282] |
