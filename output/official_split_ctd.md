# SLT-07: official-split CTD sensitivity

All official sessions; PHQ-8 >=10 including corrected session 409; unchanged 24 CTD means; median imputation and scaling fitted on train; C in 0.01/0.03/0.1/0.3/1.0 selected by dev balanced accuracy; threshold 0.5; train+dev refit for test.

| Cohort | C | Dev macro-F1 | Test macro-F1 | Test AUC |
|---|---:|---:|---:|---:|
| 107/35/47 | 1.0 | 0.782 | 0.639 | 0.654 |

Nine integrity exclusions restored, including missing interviewer turns and timestamp problems. Results do not establish measurement quality or a matched published-baseline comparison.

All 24 coordinates missing for no-pair sessions; use training-fitted feature medians. No reconstructed interviewer turns. No-pair sessions: {'train': [], 'dev': [451, 458], 'test': [480]}.

T and T+CTD cannot be evaluated for the official cohort: frozen caches, selected checkpoints and per-session predictions are unavailable.
