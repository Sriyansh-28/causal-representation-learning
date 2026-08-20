# fair_selection: aggregated results

## pehe (mean ± SD over seeds)

| condition   | condition_value       | neural_rep                | s_learner                 | t_learner                 | x_learner                 |
|:------------|:----------------------|:--------------------------|:--------------------------|:--------------------------|:--------------------------|
| scenario    | ihdp_baseline         | 1.402 ± 2.186 (med 0.717) | 4.041 ± 8.018 (med 1.237) | 2.756 ± 4.586 (med 1.091) | 3.751 ± 6.751 (med 1.284) |
| scenario    | synthetic_gamma0      | 0.578 ± 0.037 (med 0.584) | 0.434 ± 0.045 (med 0.428) | 0.887 ± 0.041 (med 0.887) | 0.431 ± 0.028 (med 0.426) |
| scenario    | synthetic_gamma2      | 0.610 ± 0.021 (med 0.604) | 0.688 ± 0.044 (med 0.683) | 0.953 ± 0.041 (med 0.955) | 0.580 ± 0.038 (med 0.583) |
| scenario    | synthetic_gamma3      | 0.623 ± 0.030 (med 0.622) | 0.771 ± 0.041 (med 0.766) | 1.005 ± 0.037 (med 0.999) | 0.628 ± 0.025 (med 0.627) |
| scenario    | synthetic_imbalance10 | 0.701 ± 0.061 (med 0.686) | 0.728 ± 0.088 (med 0.733) | 1.282 ± 0.081 (med 1.285) | 0.685 ± 0.079 (med 0.673) |

## abs_ate_error (mean ± SD over seeds)

| condition   | condition_value       | neural_rep                | s_learner                 | t_learner                 | x_learner                 |
|:------------|:----------------------|:--------------------------|:--------------------------|:--------------------------|:--------------------------|
| scenario    | ihdp_baseline         | 0.284 ± 0.583 (med 0.124) | 0.694 ± 1.626 (med 0.151) | 0.323 ± 0.544 (med 0.121) | 0.379 ± 0.446 (med 0.213) |
| scenario    | synthetic_gamma0      | 0.065 ± 0.038 (med 0.065) | 0.229 ± 0.056 (med 0.217) | 0.060 ± 0.034 (med 0.061) | 0.045 ± 0.028 (med 0.043) |
| scenario    | synthetic_gamma2      | 0.154 ± 0.055 (med 0.154) | 0.506 ± 0.048 (med 0.505) | 0.254 ± 0.075 (med 0.257) | 0.092 ± 0.060 (med 0.087) |
| scenario    | synthetic_gamma3      | 0.199 ± 0.076 (med 0.193) | 0.597 ± 0.045 (med 0.597) | 0.353 ± 0.099 (med 0.371) | 0.105 ± 0.058 (med 0.099) |
| scenario    | synthetic_imbalance10 | 0.257 ± 0.116 (med 0.247) | 0.534 ± 0.098 (med 0.545) | 0.250 ± 0.173 (med 0.233) | 0.149 ± 0.098 (med 0.143) |

## policy_regret (mean ± SD over seeds)

| condition   | condition_value       | neural_rep                | s_learner                 | t_learner                 | x_learner                 |
|:------------|:----------------------|:--------------------------|:--------------------------|:--------------------------|:--------------------------|
| scenario    | ihdp_baseline         | 0.020 ± 0.024 (med 0.011) | 0.173 ± 0.287 (med 0.065) | 0.085 ± 0.224 (med 0.017) | 0.196 ± 0.350 (med 0.073) |
| scenario    | synthetic_gamma0      | 0.008 ± 0.001 (med 0.008) | 0.008 ± 0.004 (med 0.008) | 0.100 ± 0.019 (med 0.097) | 0.019 ± 0.006 (med 0.019) |
| scenario    | synthetic_gamma2      | 0.009 ± 0.002 (med 0.008) | 0.010 ± 0.005 (med 0.009) | 0.152 ± 0.021 (med 0.154) | 0.040 ± 0.010 (med 0.037) |
| scenario    | synthetic_gamma3      | 0.010 ± 0.010 (med 0.009) | 0.011 ± 0.006 (med 0.009) | 0.183 ± 0.024 (med 0.179) | 0.047 ± 0.011 (med 0.047) |
| scenario    | synthetic_imbalance10 | 0.011 ± 0.007 (med 0.009) | 0.011 ± 0.009 (med 0.008) | 0.232 ± 0.061 (med 0.231) | 0.067 ± 0.033 (med 0.060) |

## policy_value (mean ± SD over seeds)

| condition   | condition_value       | neural_rep                   | s_learner                    | t_learner                    | x_learner                    |
|:------------|:----------------------|:-----------------------------|:-----------------------------|:-----------------------------|:-----------------------------|
| scenario    | ihdp_baseline         | 18.170 ± 20.724 (med 10.906) | 18.017 ± 20.476 (med 10.764) | 18.104 ± 20.580 (med 10.910) | 17.994 ± 20.390 (med 10.827) |
| scenario    | synthetic_gamma0      | 0.970 ± 0.062 (med 0.973)    | 0.970 ± 0.063 (med 0.975)    | 0.879 ± 0.067 (med 0.889)    | 0.960 ± 0.063 (med 0.962)    |
| scenario    | synthetic_gamma2      | 0.970 ± 0.062 (med 0.973)    | 0.969 ± 0.063 (med 0.976)    | 0.826 ± 0.062 (med 0.838)    | 0.938 ± 0.063 (med 0.945)    |
| scenario    | synthetic_gamma3      | 0.968 ± 0.062 (med 0.967)    | 0.968 ± 0.064 (med 0.970)    | 0.796 ± 0.061 (med 0.815)    | 0.931 ± 0.066 (med 0.940)    |
| scenario    | synthetic_imbalance10 | 0.968 ± 0.061 (med 0.973)    | 0.968 ± 0.065 (med 0.978)    | 0.746 ± 0.083 (med 0.749)    | 0.912 ± 0.075 (med 0.917)    |
