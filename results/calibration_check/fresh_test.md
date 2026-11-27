# Frozen recalibrator on fresh towers (E2fresh)

Written by `scripts/fig_calibration_check.py --test`. E2fresh: the E2plan2 planner with the frozen recalibrator (`boosted`) in its chance constraint, on fresh stone sets (towers 24 to 47). Intervals: 95% bootstrap over towers.

## Placements

| Placements | Prediction | n | Survival rate | Mean stated p | Brier | ECE | ECE, equal mass | Debiased L2 error | ECE at chance level | AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | Monte Carlo as stated | 184 | 0.46 | 0.39 | 0.239 [0.185, 0.293] | 0.169 [0.124, 0.256] | 0.152 [0.107, 0.242] | 0.163 [0.099, 0.279] | 0.060 | 0.73 |
| all | frozen recalibrator | 184 | 0.46 | 0.49 | 0.161 [0.133, 0.190] | 0.059 [0.050, 0.134] | 0.066 [0.043, 0.137] | 0.000 [0.000, 0.140] | 0.074 | 0.85 |
| stacked | Monte Carlo as stated | 112 | 0.37 | 0.19 | 0.236 [0.157, 0.316] | 0.196 [0.127, 0.307] | 0.176 [0.106, 0.287] | 0.170 [0.089, 0.333] | 0.067 | 0.68 |
| stacked | frozen recalibrator | 112 | 0.37 | 0.39 | 0.162 [0.135, 0.189] | 0.082 [0.063, 0.170] | 0.112 [0.066, 0.202] | 0.075 [0.000, 0.217] | 0.094 | 0.83 |
| building | Monte Carlo as stated | 48 | 0.67 | 0.60 | 0.203 [0.137, 0.271] | 0.214 [0.158, 0.308] | 0.154 [0.105, 0.284] | 0.099 [0.000, 0.350] | 0.119 | 0.79 |
| building | frozen recalibrator | 48 | 0.67 | 0.77 | 0.174 [0.111, 0.245] | 0.107 [0.068, 0.259] | 0.106 [0.067, 0.276] | 0.000 [0.000, 0.310] | 0.107 | 0.82 |
| building | planner acceptance probability | 48 | 0.67 | 0.76 | 0.185 [0.119, 0.260] | 0.103 [0.043, 0.256] | 0.112 [0.066, 0.269] | 0.000 [0.000, 0.299] | 0.088 | 0.78 |

## Whole towers

| Towers | Building placements per tower | Mean of the product of stated probabilities | Observed share of towers where every placement survived | Brier |
| --- | --- | --- | --- | --- |
| 24 | 2.00 | 0.57 [0.49, 0.64] | 0.54 [0.33, 0.75] | 0.216 |
