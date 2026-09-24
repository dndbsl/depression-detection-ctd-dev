# Exploratory timing controls (DAIC train+dev only)

135 subjects; 5 outer × 4 inner folds, 10 paired repeats. Mean ± SD across repeats; SD is not a confidence interval. No official test data used. See `docs/timing-controls-protocol.md`.

| Configuration | Macro-F1 | ROC-AUC |
|---|---:|---:|
| all24 | 0.635 ± 0.026 | 0.676 ± 0.031 |
| no_ask | 0.566 ± 0.025 | 0.610 ± 0.030 |
| latency | 0.531 ± 0.008 | 0.562 ± 0.015 |
| structure | 0.499 ± 0.020 | 0.535 ± 0.027 |
| latency_structure | 0.550 ± 0.025 | 0.576 ± 0.029 |
