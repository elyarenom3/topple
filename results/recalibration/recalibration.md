# Monte Carlo with learned recalibration

Written by `scripts/fig_recalibration.py`. Each group is predicted by models trained on the other groups only, with tower indices held out; intervals are 95% bootstrap intervals over towers and differences are paired with the stated Monte Carlo probability.

## E2all (143 placements, 24 towers)

| Method | Brier | ECE | AUC | Brier difference | ECE difference |
| --- | --- | --- | --- | --- | --- |
| Monte Carlo as stated | 0.209 [0.169, 0.253] | 0.156 [0.119, 0.229] | 0.754 [0.680, 0.823] |  |  |
| isotonic recalibration | 0.187 [0.160, 0.215] | 0.057 [0.045, 0.154] | 0.762 [0.678, 0.840] | -0.022 [-0.045, -0.001] | -0.100 [-0.148, +0.002] |
| logistic on Monte Carlo outputs | 0.189 [0.161, 0.216] | 0.057 [0.048, 0.141] | 0.745 [0.671, 0.816] | -0.021 [-0.048, +0.007] | -0.099 [-0.150, -0.008] |
| Monte Carlo with learned recalibration (final) | 0.168 [0.136, 0.203] | 0.069 [0.064, 0.169] | 0.829 [0.766, 0.884] | -0.041 [-0.081, -0.003] | -0.088 [-0.131, +0.012] |

## E4 (233 placements, 24 towers)

| Method | Brier | ECE | AUC | Brier difference | ECE difference |
| --- | --- | --- | --- | --- | --- |
| Monte Carlo as stated | 0.141 [0.109, 0.176] | 0.109 [0.084, 0.177] | 0.877 [0.828, 0.928] |  |  |
| isotonic recalibration | 0.157 [0.132, 0.182] | 0.125 [0.078, 0.192] | 0.858 [0.798, 0.917] | +0.016 [-0.012, +0.043] | +0.016 [-0.081, +0.086] |
| logistic on Monte Carlo outputs | 0.124 [0.103, 0.144] | 0.110 [0.074, 0.173] | 0.899 [0.854, 0.945] | -0.018 [-0.045, +0.009] | +0.001 [-0.080, +0.065] |
| Monte Carlo with learned recalibration (final) | 0.156 [0.116, 0.195] | 0.123 [0.081, 0.206] | 0.889 [0.849, 0.930] | +0.014 [-0.033, +0.061] | +0.014 [-0.064, +0.079] |

## E5 (819 placements, 24 towers)

| Method | Brier | ECE | AUC | Brier difference | ECE difference |
| --- | --- | --- | --- | --- | --- |
| Monte Carlo as stated | 0.255 [0.233, 0.279] | 0.180 [0.150, 0.221] | 0.673 [0.638, 0.710] |  |  |
| isotonic recalibration | 0.220 [0.205, 0.234] | 0.056 [0.035, 0.107] | 0.666 [0.629, 0.705] | -0.036 [-0.048, -0.023] | -0.124 [-0.151, -0.083] |
| logistic on Monte Carlo outputs | 0.209 [0.191, 0.229] | 0.071 [0.035, 0.121] | 0.721 [0.684, 0.759] | -0.047 [-0.061, -0.033] | -0.109 [-0.151, -0.070] |
| Monte Carlo with learned recalibration (final) | 0.184 [0.164, 0.203] | 0.074 [0.053, 0.111] | 0.802 [0.765, 0.840] | -0.072 [-0.102, -0.043] | -0.107 [-0.150, -0.059] |

## E6 (412 placements, 24 towers)

| Method | Brier | ECE | AUC | Brier difference | ECE difference |
| --- | --- | --- | --- | --- | --- |
| Monte Carlo as stated | 0.259 [0.230, 0.288] | 0.208 [0.165, 0.259] | 0.675 [0.622, 0.728] |  |  |
| isotonic recalibration | 0.219 [0.201, 0.238] | 0.066 [0.043, 0.109] | 0.675 [0.616, 0.732] | -0.040 [-0.054, -0.027] | -0.142 [-0.173, -0.099] |
| logistic on Monte Carlo outputs | 0.202 [0.186, 0.220] | 0.080 [0.053, 0.128] | 0.711 [0.680, 0.746] | -0.057 [-0.083, -0.031] | -0.127 [-0.179, -0.067] |
| Monte Carlo with learned recalibration (final) | 0.170 [0.156, 0.187] | 0.063 [0.045, 0.119] | 0.824 [0.795, 0.855] | -0.088 [-0.120, -0.059] | -0.144 [-0.194, -0.070] |

## all groups (1607 placements, 96 towers)

| Method | Brier | ECE | AUC | Brier difference | ECE difference |
| --- | --- | --- | --- | --- | --- |
| Monte Carlo as stated | 0.236 [0.218, 0.252] | 0.165 [0.140, 0.191] | 0.715 [0.687, 0.745] |  |  |
| isotonic recalibration | 0.207 [0.197, 0.218] | 0.028 [0.022, 0.064] | 0.699 [0.670, 0.732] | -0.028 [-0.037, -0.019] | -0.137 [-0.150, -0.098] |
| logistic on Monte Carlo outputs | 0.193 [0.180, 0.205] | 0.035 [0.024, 0.066] | 0.740 [0.712, 0.769] | -0.043 [-0.053, -0.031] | -0.130 [-0.147, -0.096] |
| Monte Carlo with learned recalibration (final) | 0.175 [0.161, 0.187] | 0.048 [0.037, 0.076] | 0.818 [0.795, 0.843] | -0.061 [-0.080, -0.041] | -0.117 [-0.141, -0.079] |
