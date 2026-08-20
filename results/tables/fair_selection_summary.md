# fair_selection: aggregated results

## pehe (mean ± SD over seeds)

| condition   | condition_value       | neural_rep                | s_learner                 | t_learner                 | x_learner                 |
|:------------|:----------------------|:--------------------------|:--------------------------|:--------------------------|:--------------------------|
| scenario    | ihdp_baseline         | 1.402 ± 2.186 (med 0.717) | 4.118 ± 8.204 (med 1.283) | 2.775 ± 5.047 (med 0.943) | 3.749 ± 6.846 (med 1.214) |
| scenario    | synthetic_gamma0      | 0.578 ± 0.037 (med 0.584) | 0.433 ± 0.044 (med 0.429) | 0.860 ± 0.041 (med 0.859) | 0.427 ± 0.026 (med 0.422) |
| scenario    | synthetic_gamma2      | 0.610 ± 0.021 (med 0.604) | 0.688 ± 0.044 (med 0.680) | 0.933 ± 0.045 (med 0.934) | 0.581 ± 0.036 (med 0.582) |
| scenario    | synthetic_gamma3      | 0.623 ± 0.030 (med 0.622) | 0.773 ± 0.038 (med 0.766) | 0.983 ± 0.040 (med 0.983) | 0.634 ± 0.032 (med 0.633) |
| scenario    | synthetic_imbalance10 | 0.701 ± 0.061 (med 0.686) | 0.725 ± 0.087 (med 0.748) | 1.259 ± 0.083 (med 1.266) | 0.682 ± 0.074 (med 0.670) |

## abs_ate_error (mean ± SD over seeds)

| condition   | condition_value       | neural_rep                | s_learner                 | t_learner                 | x_learner                 |
|:------------|:----------------------|:--------------------------|:--------------------------|:--------------------------|:--------------------------|
| scenario    | ihdp_baseline         | 0.284 ± 0.583 (med 0.124) | 0.730 ± 1.639 (med 0.155) | 0.354 ± 0.683 (med 0.105) | 0.368 ± 0.476 (med 0.202) |
| scenario    | synthetic_gamma0      | 0.065 ± 0.038 (med 0.065) | 0.230 ± 0.055 (med 0.223) | 0.057 ± 0.030 (med 0.055) | 0.044 ± 0.026 (med 0.043) |
| scenario    | synthetic_gamma2      | 0.154 ± 0.055 (med 0.154) | 0.507 ± 0.049 (med 0.506) | 0.246 ± 0.075 (med 0.250) | 0.090 ± 0.059 (med 0.072) |
| scenario    | synthetic_gamma3      | 0.199 ± 0.076 (med 0.193) | 0.598 ± 0.044 (med 0.595) | 0.360 ± 0.098 (med 0.378) | 0.103 ± 0.063 (med 0.097) |
| scenario    | synthetic_imbalance10 | 0.257 ± 0.116 (med 0.247) | 0.534 ± 0.096 (med 0.547) | 0.256 ± 0.172 (med 0.232) | 0.149 ± 0.092 (med 0.138) |

## policy_regret (mean ± SD over seeds)

| condition   | condition_value       | neural_rep                | s_learner                 | t_learner                 | x_learner                 |
|:------------|:----------------------|:--------------------------|:--------------------------|:--------------------------|:--------------------------|
| scenario    | ihdp_baseline         | 0.020 ± 0.024 (med 0.011) | 0.186 ± 0.367 (med 0.062) | 0.090 ± 0.215 (med 0.030) | 0.201 ± 0.345 (med 0.091) |
| scenario    | synthetic_gamma0      | 0.008 ± 0.001 (med 0.008) | 0.008 ± 0.004 (med 0.008) | 0.095 ± 0.018 (med 0.093) | 0.019 ± 0.006 (med 0.019) |
| scenario    | synthetic_gamma2      | 0.009 ± 0.002 (med 0.008) | 0.009 ± 0.006 (med 0.008) | 0.145 ± 0.021 (med 0.145) | 0.040 ± 0.011 (med 0.038) |
| scenario    | synthetic_gamma3      | 0.010 ± 0.010 (med 0.009) | 0.011 ± 0.006 (med 0.009) | 0.179 ± 0.024 (med 0.179) | 0.047 ± 0.012 (med 0.048) |
| scenario    | synthetic_imbalance10 | 0.011 ± 0.007 (med 0.009) | 0.009 ± 0.008 (med 0.007) | 0.226 ± 0.063 (med 0.221) | 0.065 ± 0.033 (med 0.056) |

## policy_value (mean ± SD over seeds)

| condition   | condition_value       | neural_rep                   | s_learner                    | t_learner                    | x_learner                    |
|:------------|:----------------------|:-----------------------------|:-----------------------------|:-----------------------------|:-----------------------------|
| scenario    | ihdp_baseline         | 18.170 ± 20.724 (med 10.906) | 18.004 ± 20.426 (med 10.779) | 18.100 ± 20.578 (med 10.892) | 17.989 ± 20.395 (med 10.830) |
| scenario    | synthetic_gamma0      | 0.970 ± 0.062 (med 0.973)    | 0.970 ± 0.063 (med 0.975)    | 0.883 ± 0.065 (med 0.890)    | 0.960 ± 0.063 (med 0.960)    |
| scenario    | synthetic_gamma2      | 0.970 ± 0.062 (med 0.973)    | 0.969 ± 0.063 (med 0.974)    | 0.833 ± 0.064 (med 0.847)    | 0.939 ± 0.062 (med 0.947)    |
| scenario    | synthetic_gamma3      | 0.968 ± 0.062 (med 0.967)    | 0.967 ± 0.064 (med 0.972)    | 0.799 ± 0.062 (med 0.821)    | 0.931 ± 0.066 (med 0.939)    |
| scenario    | synthetic_imbalance10 | 0.968 ± 0.061 (med 0.973)    | 0.969 ± 0.064 (med 0.978)    | 0.752 ± 0.085 (med 0.758)    | 0.914 ± 0.075 (med 0.918)    |
