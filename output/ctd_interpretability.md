# SLT-04: CTD interpretability

Deployed model: 24-D session-mean, L2 LogReg `C=0.3`, `class_weight='balanced'`, fit on train (n=102). 8 features are near-collinear reciprocal pairs by construction; see the pair-aggregate table before reading single-feature ranks.

## 1. Standardized coefficients (bootstrap B=2000, 95% CI)

| Feature | Group | Coef | 95% CI | Sign-consistency | Odds ratio /1SD |
|---|---|---:|---|---:|---:|
| `ask_d` | ask | -0.867 | [-1.290, -0.440] | 1.00 | 0.420 |
| `ask_bt` | ask | -0.516 | [-0.976, -0.077] | 0.99 | 0.597 |
| `ask_st` | ask | 0.429 | [-0.020, 0.994] | 0.97 | 1.536 |
| `res_h` | cross | 0.210 | [-0.322, 0.704] | 0.77 | 1.233 |
| `res_st` | res | 0.203 | [-0.289, 0.591] | 0.77 | 1.225 |
| `res_over_ask` | cross | 0.202 | [-0.370, 0.757] | 0.71 | 1.223 |
| `res_bt` | cross | -0.153 | [-0.730, 0.274] | 0.78 | 0.858 |
| `ask_sd` | ask | 0.145 | [-0.151, 0.410] | 0.84 | 1.156 |
| `ask_ud` | ask | -0.145 | [-0.410, 0.151] | 0.84 | 0.865 |
| `res_minus_ask` | cross | 0.134 | [-0.052, 0.341] | 0.92 | 1.144 |
| `ask_minus_res` | cross | -0.134 | [-0.341, 0.052] | 0.92 | 0.874 |
| `ask_ds` | ask | -0.111 | [-0.417, 0.162] | 0.84 | 0.895 |
| `ask_us` | ask | -0.111 | [-0.417, 0.162] | 0.84 | 0.895 |
| `ask_du` | ask | -0.111 | [-0.588, 0.197] | 0.75 | 0.895 |
| `ask_su` | ask | -0.111 | [-0.588, 0.197] | 0.75 | 0.895 |
| `ask_over_res` | cross | -0.088 | [-0.628, 0.384] | 0.69 | 0.916 |
| `res_sd` | res | -0.087 | [-0.424, 0.304] | 0.63 | 0.917 |
| `res_ud` | res | 0.087 | [-0.304, 0.424] | 0.63 | 1.090 |
| `res_du` | res | 0.076 | [-0.444, 0.413] | 0.60 | 1.079 |
| `res_su` | res | 0.076 | [-0.444, 0.413] | 0.60 | 1.079 |
| `res_us` | res | 0.048 | [-0.303, 0.325] | 0.58 | 1.049 |
| `res_ds` | res | 0.048 | [-0.303, 0.325] | 0.58 | 1.049 |
| `duration_sum` | cross | -0.047 | [-0.216, 0.133] | 0.67 | 0.954 |
| `res_d` | res | 0.042 | [-0.130, 0.233] | 0.71 | 1.043 |

## 2. Permutation importance on dev (f1_macro drop, 100 repeats)

| Feature | Mean importance | Std |
|---|---:|---:|
| `ask_d` | 0.2429 | 0.0642 |
| `res_h` | 0.0479 | 0.0467 |
| `res_st` | 0.0408 | 0.0486 |
| `ask_du` | 0.0347 | 0.0353 |
| `ask_su` | 0.0347 | 0.0353 |
| `res_over_ask` | 0.0301 | 0.0398 |
| `ask_bt` | 0.0241 | 0.0611 |
| `res_du` | 0.0206 | 0.0192 |
| `res_su` | 0.0206 | 0.0192 |
| `res_bt` | 0.0156 | 0.0347 |
| `res_ds` | 0.0014 | 0.0061 |
| `res_us` | 0.0014 | 0.0061 |
| `res_minus_ask` | -0.0031 | 0.0298 |
| `ask_minus_res` | -0.0031 | 0.0298 |
| `duration_sum` | -0.0032 | 0.0216 |
| `res_d` | -0.0046 | 0.0119 |
| `ask_ds` | -0.0067 | 0.0281 |
| `ask_us` | -0.0067 | 0.0281 |
| `ask_over_res` | -0.0096 | 0.0284 |
| `res_ud` | -0.0123 | 0.0181 |
| `res_sd` | -0.0123 | 0.0181 |
| `ask_st` | -0.0478 | 0.0521 |
| `ask_ud` | -0.0706 | 0.0244 |
| `ask_sd` | -0.0706 | 0.0244 |

## 3. Univariate direction consistency (point-biserial r, single-feature AUC)

| Feature | train r | dev r | test r | train AUC | dev AUC | test AUC | Sign-consistent |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ask_d` | -0.220 | -0.412 | -0.239 | 0.365 | 0.278 | 0.387 | yes |
| `res_d` | 0.049 | 0.289 | -0.134 | 0.539 | 0.631 | 0.355 | no |
| `res_minus_ask` | 0.073 | 0.328 | -0.113 | 0.553 | 0.663 | 0.387 | no |
| `ask_minus_res` | -0.073 | -0.328 | 0.113 | 0.447 | 0.337 | 0.613 | no |
| `duration_sum` | 0.025 | 0.248 | -0.155 | 0.528 | 0.611 | 0.329 | no |
| `res_over_ask` | 0.107 | 0.441 | 0.042 | 0.585 | 0.758 | 0.438 | yes |
| `ask_over_res` | -0.053 | -0.155 | -0.143 | 0.454 | 0.413 | 0.415 | yes |
| `ask_ud` | -0.060 | 0.348 | 0.020 | 0.506 | 0.726 | 0.512 | no |
| `ask_du` | 0.032 | -0.359 | 0.112 | 0.497 | 0.286 | 0.521 | no |
| `res_ud` | 0.057 | -0.173 | -0.272 | 0.537 | 0.413 | 0.332 | no |
| `res_du` | -0.041 | 0.224 | 0.236 | 0.444 | 0.643 | 0.666 | no |
| `ask_sd` | 0.060 | -0.348 | -0.020 | 0.494 | 0.274 | 0.488 | no |
| `ask_ds` | -0.029 | -0.266 | -0.080 | 0.442 | 0.365 | 0.433 | yes |
| `res_sd` | -0.057 | 0.173 | 0.272 | 0.463 | 0.587 | 0.668 | no |
| `res_ds` | 0.024 | -0.189 | -0.287 | 0.533 | 0.405 | 0.300 | no |
| `ask_su` | 0.032 | -0.359 | 0.112 | 0.497 | 0.286 | 0.521 | no |
| `ask_us` | -0.029 | -0.266 | -0.080 | 0.442 | 0.365 | 0.433 | yes |
| `res_su` | -0.041 | 0.224 | 0.236 | 0.444 | 0.643 | 0.666 | no |
| `res_us` | 0.024 | -0.189 | -0.287 | 0.533 | 0.405 | 0.300 | no |
| `res_h` | 0.104 | 0.296 | 0.528 | 0.558 | 0.635 | 0.829 | yes |
| `ask_bt` | -0.124 | 0.258 | 0.012 | 0.432 | 0.694 | 0.517 | no |
| `res_bt` | -0.130 | -0.196 | -0.257 | 0.356 | 0.359 | 0.342 | yes |
| `ask_st` | 0.085 | -0.374 | -0.060 | 0.523 | 0.270 | 0.462 | no |
| `res_st` | 0.026 | 0.319 | -0.047 | 0.481 | 0.651 | 0.429 | no |

## 4. Pair-level aggregate importance (reciprocal pairs)

| Pair | Sum |coef| |
|---|---:|
| `ask_ud+ask_du` | 0.256 |
| `ask_sd+ask_ds` | 0.256 |
| `ask_su+ask_us` | 0.222 |
| `res_ud+res_du` | 0.163 |
| `res_sd+res_ds` | 0.134 |
| `res_su+res_us` | 0.124 |
| `res_over_ask+ask_over_res` | 0.290 |
| `res_minus_ask+ask_minus_res` | 0.269 |

Unpaired features: `ask_d`, `res_d`, `duration_sum`, `res_h`, `ask_bt`, `res_bt`, `ask_st`, `res_st`


## 5. Collinearity diagnostic

**Exact linear dependencies found (|r| > 0.999)** -- these are algebraic identities at the session-mean level, not just high correlation, and go beyond the 8 documented reciprocal pairs:

| Pair | r |
|---|---:|
| `res_minus_ask` / `ask_minus_res` | -1.0000 |
| `ask_ud` / `ask_sd` | -1.0000 |
| `ask_du` / `ask_su` | 1.0000 |
| `res_ud` / `res_sd` | -1.0000 |
| `res_du` / `res_su` | 1.0000 |
| `ask_ds` / `ask_us` | 1.0000 |
| `res_ds` / `res_us` | 1.0000 |

> VIF is numerically unstable / not meaningful here: the design matrix is rank-deficient because of the exact linear dependencies listed in exact_linear_dependencies (not just high correlation). Use the correlation matrix and pair-level importance below instead of VIF magnitudes.


Full correlation matrix is in `ctd_interpretability.json` (`collinearity.correlation_matrix`).
