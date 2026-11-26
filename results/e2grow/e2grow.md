# E2 with a looser chance constraint

Written by `scripts/fig_e2grow.py`. E2all uses delta 0.05 and E2grow delta 0.3; both use the perceived stones and the same stones per tower index. Intervals: 95% bootstrap over towers.

| Experiment | Towers | Stones per tower (mean, max) | Height cm (mean, max) | Collapse rate | Placements | Stacked placements |
| --- | --- | --- | --- | --- | --- | --- |
| E2all | 24 | 1.04, 2 | 6.5, 12.5 | 0.00 | 143 | 71 |
| E2grow | 24 | 1.62, 3 | 10.2, 21.1 | 0.00 | 179 | 107 |

## Placements: all

| Experiment | Prediction | n | Survival rate | Mean stated p | Brier | ECE | AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E2all | Monte Carlo | 143 | 0.46 | 0.45 | 0.209 [0.171, 0.254] | 0.156 [0.120, 0.228] | 0.75 |
| E2all | point estimate | 143 | 0.46 | 0.75 | 0.441 [0.376, 0.500] | 0.441 [0.376, 0.500] | 0.58 |
| E2all | rocking margin, Monte Carlo | 143 | 0.46 | 0.40 | 0.222 [0.178, 0.272] | 0.182 [0.154, 0.256] | 0.74 |
| E2all | Monte Carlo with learned recalibration | 143 | 0.46 | 0.46 | 0.168 [0.137, 0.201] | 0.069 [0.064, 0.166] | 0.83 |
| E2grow | Monte Carlo | 179 | 0.39 | 0.40 | 0.188 [0.158, 0.221] | 0.103 [0.080, 0.179] | 0.76 |
| E2grow | point estimate | 179 | 0.39 | 0.66 | 0.425 [0.371, 0.473] | 0.425 [0.371, 0.473] | 0.62 |
| E2grow | rocking margin, Monte Carlo | 179 | 0.39 | 0.35 | 0.204 [0.163, 0.248] | 0.165 [0.128, 0.239] | 0.74 |
| E2grow | Monte Carlo with learned recalibration | 179 | 0.39 | 0.37 | 0.163 [0.129, 0.200] | 0.054 [0.042, 0.137] | 0.81 |

## Placements: ground

| Experiment | Prediction | n | Survival rate | Mean stated p | Brier | ECE | AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E2all | Monte Carlo | 72 | 0.56 | 0.67 | 0.174 [0.128, 0.223] | 0.141 [0.107, 0.248] | 0.86 |
| E2all | point estimate | 72 | 0.56 | 0.93 | 0.375 [0.278, 0.458] | 0.375 [0.278, 0.458] | 0.58 |
| E2all | rocking margin, Monte Carlo | 72 | 0.56 | 0.64 | 0.171 [0.116, 0.224] | 0.136 [0.106, 0.244] | 0.84 |
| E2all | Monte Carlo with learned recalibration | 72 | 0.56 | 0.59 | 0.129 [0.079, 0.182] | 0.084 [0.065, 0.185] | 0.90 |
| E2grow | Monte Carlo | 72 | 0.54 | 0.65 | 0.175 [0.131, 0.224] | 0.134 [0.099, 0.255] | 0.85 |
| E2grow | point estimate | 72 | 0.54 | 0.93 | 0.389 [0.292, 0.486] | 0.389 [0.292, 0.486] | 0.58 |
| E2grow | rocking margin, Monte Carlo | 72 | 0.54 | 0.63 | 0.182 [0.124, 0.240] | 0.161 [0.122, 0.270] | 0.83 |
| E2grow | Monte Carlo with learned recalibration | 72 | 0.54 | 0.54 | 0.133 [0.082, 0.186] | 0.092 [0.070, 0.204] | 0.89 |

## Placements: stacked

| Experiment | Prediction | n | Survival rate | Mean stated p | Brier | ECE | AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E2all | Monte Carlo | 71 | 0.37 | 0.22 | 0.245 [0.176, 0.324] | 0.190 [0.134, 0.315] | 0.66 |
| E2all | point estimate | 71 | 0.37 | 0.56 | 0.507 [0.392, 0.623] | 0.507 [0.392, 0.623] | 0.51 |
| E2all | rocking margin, Monte Carlo | 71 | 0.37 | 0.15 | 0.274 [0.191, 0.359] | 0.288 [0.200, 0.385] | 0.65 |
| E2all | Monte Carlo with learned recalibration | 71 | 0.37 | 0.33 | 0.207 [0.167, 0.247] | 0.090 [0.069, 0.222] | 0.69 |
| E2grow | Monte Carlo | 107 | 0.28 | 0.22 | 0.196 [0.144, 0.257] | 0.101 [0.074, 0.218] | 0.66 |
| E2grow | point estimate | 107 | 0.28 | 0.49 | 0.449 [0.363, 0.526] | 0.449 [0.363, 0.526] | 0.56 |
| E2grow | rocking margin, Monte Carlo | 107 | 0.28 | 0.16 | 0.220 [0.152, 0.298] | 0.197 [0.141, 0.304] | 0.64 |
| E2grow | Monte Carlo with learned recalibration | 107 | 0.28 | 0.25 | 0.183 [0.141, 0.232] | 0.056 [0.044, 0.171] | 0.70 |
