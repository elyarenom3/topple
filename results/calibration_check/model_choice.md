# Recalibrator choice

Written by `scripts/fig_calibration_check.py --fit`. 2660 placements from E2all, E2cand, E2grow, E2plan, E2plan2, E2view, E4, E5, E6 (towers 0 to 23), leave one tower out. Intervals: 95% bootstrap over tower indices.

## Leave one tower out

| Placements | Prediction | n | Survival rate | Mean stated p | Brier | ECE | ECE, equal mass | Debiased L2 error | ECE at chance level | AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | Monte Carlo as stated | 2660 | 0.47 | 0.43 | 0.230 [0.214, 0.249] | 0.151 [0.130, 0.175] | 0.141 [0.126, 0.170] | 0.169 [0.145, 0.198] | 0.017 | 0.73 |
| all | gradient boosted (current) | 2660 | 0.47 | 0.47 | 0.167 [0.153, 0.180] | 0.033 [0.025, 0.057] | 0.023 [0.017, 0.052] | 0.025 [0.007, 0.059] | 0.019 | 0.83 |
| all | logistic on logit p, stacked terms | 2660 | 0.47 | 0.47 | 0.208 [0.197, 0.219] | 0.039 [0.030, 0.066] | 0.054 [0.037, 0.077] | 0.053 [0.037, 0.084] | 0.019 | 0.71 |
| stacked | Monte Carlo as stated | 1412 | 0.35 | 0.21 | 0.254 [0.223, 0.288] | 0.171 [0.133, 0.218] | 0.171 [0.132, 0.216] | 0.185 [0.143, 0.236] | 0.021 | 0.61 |
| stacked | gradient boosted (current) | 1412 | 0.35 | 0.35 | 0.199 [0.185, 0.213] | 0.049 [0.033, 0.082] | 0.052 [0.030, 0.080] | 0.054 [0.024, 0.092] | 0.026 | 0.72 |
| stacked | logistic on logit p, stacked terms | 1412 | 0.35 | 0.35 | 0.222 [0.207, 0.237] | 0.012 [0.011, 0.060] | 0.026 [0.022, 0.070] | 0.000 [0.000, 0.074] | 0.019 | 0.60 |

Frozen: **gradient boosted (current)** (`boosted`), fitted on all 2660 placements, saved to `results/recalibration/frozen.pkl`. The simpler model is kept unless the boosted one lowers the Brier score by more than 0.002.
