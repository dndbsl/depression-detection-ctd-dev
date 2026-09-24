# Manuscript number ledger

Exact display strings; full precision, source hashes, keys, manuscript lines and figure paths are in `paper_numbers.json`. Only used TeX macros are emitted. Figure labels use the same display strings; other ledger entries preserve audited checks.

| Manuscript number key | Display value | Source file → key | Command / provenance | Used in current paper |
|---|---|---|---|---|
| `trainN` | 102 | `output/daic_gate_results.json` → `cohort.train.n` | `python src/ctd/ml_splits.py (copied aggregate gate artifact)` | TeX lines 38 |
| `trainPositive` | 29 | `output/daic_gate_results.json` → `cohort.train.n_depressed` | `python src/ctd/ml_splits.py (copied aggregate gate artifact)` | TeX lines 38 |
| `devN` | 33 | `output/daic_gate_results.json` → `cohort.dev.n` | `python src/ctd/ml_splits.py (copied aggregate gate artifact)` | TeX lines 38 |
| `devPositive` | 12 | `output/daic_gate_results.json` → `cohort.dev.n_depressed` | `python src/ctd/ml_splits.py (copied aggregate gate artifact)` | TeX lines 38 |
| `testN` | 45 | `output/daic_gate_results.json` → `cohort.test.n` | `python src/ctd/ml_splits.py (copied aggregate gate artifact)` | TeX lines 38 |
| `testPositive` | 14 | `output/daic_gate_results.json` → `cohort.test.n_depressed` | `python src/ctd/ml_splits.py (copied aggregate gate artifact)` | TeX lines 38 |
| `cleanN` | 180 | `output/daic_gate_results.json` → `sum(cohort.*.n)` | `python src/ctd/ml_splits.py (copied aggregate gate artifact)` | TeX lines 132 |
| `fitC` | 0.3 | `output/daic_gate_results.json` → `session_mean_24d.models.logreg_l2.selected_hyperparam` | `python src/ctd/ml_splits.py (copied aggregate gate artifact)` | TeX lines 51 |
| `gateDev` | 0.746154 | `output/daic_gate_results.json` → `session_mean_24d.models.logreg_l2.dev.macro_f1` | `python src/ctd/ml_splits.py (copied aggregate gate artifact)` | audit only |
| `gateTest` | 0.630946 | `output/daic_gate_results.json` → `session_mean_24d.models.logreg_l2.test.macro_f1.point` | `python src/ctd/ml_splits.py (copied aggregate gate artifact)` | audit only |
| `functionalDev` | 0.659 | `output/daic_gate_results.json` → `functionals_240d.models.logreg_l1.dev.macro_f1` | `python src/ctd/ml_splits.py (copied aggregate gate artifact)` | audit only |
| `functionalTest` | 0.583 | `output/daic_gate_results.json` → `functionals_240d.models.logreg_l1.test.macro_f1.point` | `python src/ctd/ml_splits.py (copied aggregate gate artifact)` | audit only |
| `nestedN` | 135 | `output/timing_controls.json` → `n_subjects` | `python src/ctd/timing_controls.py` | TeX lines 49, 71 |
| `all24Fmean` | 0.635 | `output/timing_controls.json` → `results.all24.macro_f1.mean` | `python src/ctd/timing_controls.py` | TeX lines 20, 66; paper/figures/timing-controls.pdf |
| `all24Fsd` | 0.026 | `output/timing_controls.json` → `results.all24.macro_f1.sd` | `python src/ctd/timing_controls.py` | paper/figures/timing-controls.pdf |
| `all24Amean` | 0.676 | `output/timing_controls.json` → `results.all24.roc_auc.mean` | `python src/ctd/timing_controls.py` | TeX lines 66; paper/figures/timing-controls.pdf |
| `all24Asd` | 0.031 | `output/timing_controls.json` → `results.all24.roc_auc.sd` | `python src/ctd/timing_controls.py` | paper/figures/timing-controls.pdf |
| `no_askFmean` | 0.566 | `output/timing_controls.json` → `results.no_ask.macro_f1.mean` | `python src/ctd/timing_controls.py` | TeX lines 20, 66; paper/figures/timing-controls.pdf |
| `no_askFsd` | 0.025 | `output/timing_controls.json` → `results.no_ask.macro_f1.sd` | `python src/ctd/timing_controls.py` | paper/figures/timing-controls.pdf |
| `no_askAmean` | 0.610 | `output/timing_controls.json` → `results.no_ask.roc_auc.mean` | `python src/ctd/timing_controls.py` | TeX lines 66; paper/figures/timing-controls.pdf |
| `no_askAsd` | 0.030 | `output/timing_controls.json` → `results.no_ask.roc_auc.sd` | `python src/ctd/timing_controls.py` | paper/figures/timing-controls.pdf |
| `latencyFmean` | 0.531 | `output/timing_controls.json` → `results.latency.macro_f1.mean` | `python src/ctd/timing_controls.py` | paper/figures/timing-controls.pdf |
| `latencyFsd` | 0.008 | `output/timing_controls.json` → `results.latency.macro_f1.sd` | `python src/ctd/timing_controls.py` | paper/figures/timing-controls.pdf |
| `latencyAmean` | 0.562 | `output/timing_controls.json` → `results.latency.roc_auc.mean` | `python src/ctd/timing_controls.py` | paper/figures/timing-controls.pdf |
| `latencyAsd` | 0.015 | `output/timing_controls.json` → `results.latency.roc_auc.sd` | `python src/ctd/timing_controls.py` | paper/figures/timing-controls.pdf |
| `structureFmean` | 0.499 | `output/timing_controls.json` → `results.structure.macro_f1.mean` | `python src/ctd/timing_controls.py` | paper/figures/timing-controls.pdf |
| `structureFsd` | 0.020 | `output/timing_controls.json` → `results.structure.macro_f1.sd` | `python src/ctd/timing_controls.py` | paper/figures/timing-controls.pdf |
| `structureAmean` | 0.535 | `output/timing_controls.json` → `results.structure.roc_auc.mean` | `python src/ctd/timing_controls.py` | paper/figures/timing-controls.pdf |
| `structureAsd` | 0.027 | `output/timing_controls.json` → `results.structure.roc_auc.sd` | `python src/ctd/timing_controls.py` | paper/figures/timing-controls.pdf |
| `latency_structureFmean` | 0.550 | `output/timing_controls.json` → `results.latency_structure.macro_f1.mean` | `python src/ctd/timing_controls.py` | TeX lines 20, 66; paper/figures/timing-controls.pdf |
| `latency_structureFsd` | 0.025 | `output/timing_controls.json` → `results.latency_structure.macro_f1.sd` | `python src/ctd/timing_controls.py` | paper/figures/timing-controls.pdf |
| `latency_structureAmean` | 0.576 | `output/timing_controls.json` → `results.latency_structure.roc_auc.mean` | `python src/ctd/timing_controls.py` | paper/figures/timing-controls.pdf |
| `latency_structureAsd` | 0.029 | `output/timing_controls.json` → `results.latency_structure.roc_auc.sd` | `python src/ctd/timing_controls.py` | paper/figures/timing-controls.pdf |
| `nestedRepeats` | 10 | `output/timing_controls.json` → `results.all24.n_repeats` | `python src/ctd/timing_controls.py` | TeX lines 49, 71 |
| `all24Dim` | 24 | `output/ctd_ablation_groups.json` → `configs.all24.n_features` | `python src/ctd/ablation_groups.py` | TeX lines 31, 87 |
| `all24C` | 0.30 | `output/ctd_ablation_groups.json` → `configs.all24.selected_C` | `python src/ctd/ablation_groups.py` | TeX lines 87 |
| `all24Dev` | 0.746 | `output/ctd_ablation_groups.json` → `configs.all24.dev.macro_f1` | `python src/ctd/ablation_groups.py` | TeX lines 87 |
| `all24Test` | 0.631 | `output/ctd_ablation_groups.json` → `configs.all24.test.macro_f1.point` | `python src/ctd/ablation_groups.py` | TeX lines 87 |
| `no_askDim` | 15 | `output/ctd_ablation_groups.json` → `configs.no_ask.n_features` | `python src/ctd/ablation_groups.py` | TeX lines 88 |
| `no_askC` | 1.00 | `output/ctd_ablation_groups.json` → `configs.no_ask.selected_C` | `python src/ctd/ablation_groups.py` | TeX lines 88 |
| `no_askDev` | 0.814 | `output/ctd_ablation_groups.json` → `configs.no_ask.dev.macro_f1` | `python src/ctd/ablation_groups.py` | TeX lines 75, 88 |
| `no_askTest` | 0.641 | `output/ctd_ablation_groups.json` → `configs.no_ask.test.macro_f1.point` | `python src/ctd/ablation_groups.py` | TeX lines 75, 88 |
| `res_onlyDim` | 8 | `output/ctd_ablation_groups.json` → `configs.res_only.n_features` | `python src/ctd/ablation_groups.py` | TeX lines 45, 90 |
| `res_onlyC` | 1.00 | `output/ctd_ablation_groups.json` → `configs.res_only.selected_C` | `python src/ctd/ablation_groups.py` | TeX lines 90 |
| `res_onlyDev` | 0.607 | `output/ctd_ablation_groups.json` → `configs.res_only.dev.macro_f1` | `python src/ctd/ablation_groups.py` | TeX lines 90 |
| `res_onlyTest` | 0.517 | `output/ctd_ablation_groups.json` → `configs.res_only.test.macro_f1.point` | `python src/ctd/ablation_groups.py` | TeX lines 90 |
| `ask_onlyDim` | 9 | `output/ctd_ablation_groups.json` → `configs.ask_only.n_features` | `python src/ctd/ablation_groups.py` | TeX lines 45, 91 |
| `ask_onlyC` | 0.01 | `output/ctd_ablation_groups.json` → `configs.ask_only.selected_C` | `python src/ctd/ablation_groups.py` | TeX lines 91 |
| `ask_onlyDev` | 0.594 | `output/ctd_ablation_groups.json` → `configs.ask_only.dev.macro_f1` | `python src/ctd/ablation_groups.py` | TeX lines 91 |
| `ask_onlyTest` | 0.593 | `output/ctd_ablation_groups.json` → `configs.ask_only.test.macro_f1.point` | `python src/ctd/ablation_groups.py` | TeX lines 91 |
| `no_crossDim` | 17 | `output/ctd_ablation_groups.json` → `configs.no_cross.n_features` | `python src/ctd/ablation_groups.py` | TeX lines 89 |
| `no_crossC` | 0.10 | `output/ctd_ablation_groups.json` → `configs.no_cross.selected_C` | `python src/ctd/ablation_groups.py` | TeX lines 89 |
| `no_crossDev` | 0.700 | `output/ctd_ablation_groups.json` → `configs.no_cross.dev.macro_f1` | `python src/ctd/ablation_groups.py` | TeX lines 89 |
| `no_crossTest` | 0.527 | `output/ctd_ablation_groups.json` → `configs.no_cross.test.macro_f1.point` | `python src/ctd/ablation_groups.py` | TeX lines 89 |
| `ablationdevpoint` | -0.068 | `output/ctd_ablation_groups.json` → `primary_ablation_paired_bootstrap.dev.point` | `python src/ctd/ablation_groups.py` | TeX lines 75 |
| `ablationdevci_low` | -0.216 | `output/ctd_ablation_groups.json` → `primary_ablation_paired_bootstrap.dev.ci_low` | `python src/ctd/ablation_groups.py` | TeX lines 75 |
| `ablationdevci_high` | +0.083 | `output/ctd_ablation_groups.json` → `primary_ablation_paired_bootstrap.dev.ci_high` | `python src/ctd/ablation_groups.py` | TeX lines 75 |
| `ablationtestpoint` | -0.010 | `output/ctd_ablation_groups.json` → `primary_ablation_paired_bootstrap.test.point` | `python src/ctd/ablation_groups.py` | TeX lines 75 |
| `ablationtestci_low` | -0.089 | `output/ctd_ablation_groups.json` → `primary_ablation_paired_bootstrap.test.ci_low` | `python src/ctd/ablation_groups.py` | TeX lines 75 |
| `ablationtestci_high` | +0.054 | `output/ctd_ablation_groups.json` → `primary_ablation_paired_bootstrap.test.ci_high` | `python src/ctd/ablation_groups.py` | TeX lines 75 |
| `ask_dcoef` | -0.867 | `output/ctd_interpretability.json` → `coefficients.ask_d.coef` | `python src/ctd/interpret.py` | TeX lines 96 |
| `ask_dci_low` | -1.290 | `output/ctd_interpretability.json` → `coefficients.ask_d.ci_low` | `python src/ctd/interpret.py` | TeX lines 96 |
| `ask_dci_high` | -0.440 | `output/ctd_interpretability.json` → `coefficients.ask_d.ci_high` | `python src/ctd/interpret.py` | TeX lines 96 |
| `res_hcoef` | +0.210 | `output/ctd_interpretability.json` → `coefficients.res_h.coef` | `python src/ctd/interpret.py` | TeX lines 96 |
| `res_hci_low` | -0.322 | `output/ctd_interpretability.json` → `coefficients.res_h.ci_low` | `python src/ctd/interpret.py` | TeX lines 96 |
| `res_hci_high` | +0.704 | `output/ctd_interpretability.json` → `coefficients.res_h.ci_high` | `python src/ctd/interpret.py` | TeX lines 96 |
| `AThreshold` | 0.65 | `output/fusion_wavlm_seed44.json` → `results.single_wavlm.threshold` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | TeX lines 115 |
| `ADev` | 0.673 | `output/fusion_wavlm_seed44.json` → `results.single_wavlm.dev.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | TeX lines 115 |
| `ATestpoint` | 0.545 | `output/fusion_wavlm_seed44.json` → `results.single_wavlm.test.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | TeX lines 115 |
| `ATestci_low` | 0.392 | `output/fusion_wavlm_seed44.json` → `results.single_wavlm.test.macro_f1.ci_low` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | TeX lines 115 |
| `ATestci_high` | 0.681 | `output/fusion_wavlm_seed44.json` → `results.single_wavlm.test.macro_f1.ci_high` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | TeX lines 115 |
| `TThreshold` | 0.5238 | `output/fusion_wavlm_seed44.json` → `results.single_roberta.threshold` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | TeX lines 112 |
| `TDev` | 0.690 | `output/fusion_wavlm_seed44.json` → `results.single_roberta.dev.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | TeX lines 112 |
| `TTestpoint` | 0.631 | `output/fusion_wavlm_seed44.json` → `results.single_roberta.test.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | TeX lines 112 |
| `TTestci_low` | 0.474 | `output/fusion_wavlm_seed44.json` → `results.single_roberta.test.macro_f1.ci_low` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | TeX lines 112 |
| `TTestci_high` | 0.769 | `output/fusion_wavlm_seed44.json` → `results.single_roberta.test.macro_f1.ci_high` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | TeX lines 112 |
| `CTDThreshold` | 0.50 | `output/fusion_wavlm_seed44.json` → `results.single_ctd.threshold` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | TeX lines 111 |
| `CTDDev` | 0.746 | `output/fusion_wavlm_seed44.json` → `results.single_ctd.dev.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | TeX lines 111 |
| `CTDTestpoint` | 0.631 | `output/fusion_wavlm_seed44.json` → `results.single_ctd.test.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | TeX lines 99, 111 |
| `CTDTestci_low` | 0.472 | `output/fusion_wavlm_seed44.json` → `results.single_ctd.test.macro_f1.ci_low` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | TeX lines 111 |
| `CTDTestci_high` | 0.771 | `output/fusion_wavlm_seed44.json` → `results.single_ctd.test.macro_f1.ci_high` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | TeX lines 111 |
| `ATThreshold` | 0.60 | `output/fusion_wavlm_seed44.json` → `results.wconvex[wavlm+roberta].threshold` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | audit only |
| `ATDev` | 0.700 | `output/fusion_wavlm_seed44.json` → `results.wconvex[wavlm+roberta].dev.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | audit only |
| `ATTestpoint` | 0.568 | `output/fusion_wavlm_seed44.json` → `results.wconvex[wavlm+roberta].test.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | audit only |
| `ATTestci_low` | 0.410 | `output/fusion_wavlm_seed44.json` → `results.wconvex[wavlm+roberta].test.macro_f1.ci_low` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | audit only |
| `ATTestci_high` | 0.709 | `output/fusion_wavlm_seed44.json` → `results.wconvex[wavlm+roberta].test.macro_f1.ci_high` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | audit only |
| `ACTDThreshold` | 0.50 | `output/fusion_wavlm_seed44.json` → `results.wconvex[wavlm+ctd].threshold` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | audit only |
| `ACTDDev` | 0.775 | `output/fusion_wavlm_seed44.json` → `results.wconvex[wavlm+ctd].dev.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | audit only |
| `ACTDTestpoint` | 0.612 | `output/fusion_wavlm_seed44.json` → `results.wconvex[wavlm+ctd].test.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | audit only |
| `ACTDTestci_low` | 0.458 | `output/fusion_wavlm_seed44.json` → `results.wconvex[wavlm+ctd].test.macro_f1.ci_low` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | audit only |
| `ACTDTestci_high` | 0.750 | `output/fusion_wavlm_seed44.json` → `results.wconvex[wavlm+ctd].test.macro_f1.ci_high` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | audit only |
| `TCTDThreshold` | 0.55 | `output/fusion_wavlm_seed44.json` → `results.wconvex[roberta+ctd].threshold` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | TeX lines 113 |
| `TCTDDev` | 0.804 | `output/fusion_wavlm_seed44.json` → `results.wconvex[roberta+ctd].dev.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | TeX lines 113 |
| `TCTDTestpoint` | 0.669 | `output/fusion_wavlm_seed44.json` → `results.wconvex[roberta+ctd].test.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | TeX lines 99, 113 |
| `TCTDTestci_low` | 0.509 | `output/fusion_wavlm_seed44.json` → `results.wconvex[roberta+ctd].test.macro_f1.ci_low` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | TeX lines 113 |
| `TCTDTestci_high` | 0.806 | `output/fusion_wavlm_seed44.json` → `results.wconvex[roberta+ctd].test.macro_f1.ci_high` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | TeX lines 113 |
| `fusionDelta` | 0.038 | `output/fusion_wavlm_seed44.json` → `results.wconvex[roberta+ctd].test.macro_f1.point - results.single_roberta.test.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | TeX lines 99 |
| `robertaWeight` | 0.3 | `output/fusion_wavlm_seed44.json` → `results.wconvex[roberta+ctd].weights.roberta` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | TeX lines 122 |
| `ctdWeight` | 0.7 | `output/fusion_wavlm_seed44.json` → `results.wconvex[roberta+ctd].weights.ctd` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | TeX lines 122 |
| `acousticWeight` | 0.0 | `output/fusion_wavlm_seed44.json` → `results.wconvex[wavlm+roberta+ctd].weights.wavlm` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | TeX lines 122 |
| `meandev` | 0.738 | `output/mean_prob_seed44.json` → `mean_prob[roberta+ctd].dev.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | TeX lines 99, 114 |
| `meantest` | 0.650 | `output/mean_prob_seed44.json` → `mean_prob[roberta+ctd].test.macro_f1.point` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | TeX lines 99, 114 |
| `meanThreshold` | 0.55 | `output/mean_prob_seed44.json` → `mean_prob[roberta+ctd].threshold` | `recorded artifact at 49520dc; original run_fusion_wavlm_seeds.py --wavlm-seeds 44; not rerun` | TeX lines 114 |
| `pdchN` | 62 | `output/pdch_ctd_results.json` → `labels.n_sessions` | `python src/ctd/pdch_experiment.py` | TeX lines 40 |
| `pdchSubjects` | 46 | `output/pdch_ctd_results.json` → `labels.n_subjects` | `python src/ctd/pdch_experiment.py` | TeX lines 40 |
| `pdchRepeated` | 16 | `output/pdch_ctd_results.json` → `labels.n_subjects_with_two_sessions` | `python src/ctd/pdch_experiment.py` | TeX lines 40 |
| `pdchPositive` | 27 | `output/pdch_ctd_results.json` → `labels.n_positive` | `python src/ctd/pdch_experiment.py` | TeX lines 40 |
| `pdchFmean` | 0.520 | `output/pdch_ctd_results.json` → `classification.all24.macro_f1.mean` | `python src/ctd/pdch_experiment.py` | TeX lines 128 |
| `pdchFsd` | 0.055 | `output/pdch_ctd_results.json` → `classification.all24.macro_f1.sd` | `python src/ctd/pdch_experiment.py` | TeX lines 128 |
| `pdchAmean` | 0.476 | `output/pdch_ctd_results.json` → `classification.all24.roc_auc.mean` | `python src/ctd/pdch_experiment.py` | TeX lines 128 |
| `pdchAsd` | 0.066 | `output/pdch_ctd_results.json` → `classification.all24.roc_auc.sd` | `python src/ctd/pdch_experiment.py` | TeX lines 128 |
| `pdchRepeats` | 20 | `output/pdch_ctd_results.json` → `classification.all24.n_repeats` | `python src/ctd/pdch_experiment.py` | TeX lines 59 |
| `daicLatencyr` | 0.245 | `output/shared_signal_audit.json` → `E4_secondary_targets.F2.daic.per_feature.res_h__amean.r` | `python src/ctd/shared_signal_audit.py` | TeX lines 128 |
| `daicLatencyp` | 0.004 | `output/shared_signal_audit.json` → `E4_secondary_targets.F2.daic.per_feature.res_h__amean.p_perm` | `python src/ctd/shared_signal_audit.py` | audit only |
| `daicLatencyq` | 0.051 | `output/shared_signal_audit.json` → `E4_secondary_targets.F2.daic.per_feature.res_h__amean.q_bh` | `python src/ctd/shared_signal_audit.py` | TeX lines 128 |
| `pdchLatencyr` | 0.267 | `output/shared_signal_audit.json` → `E4_secondary_targets.F2.pdch.per_feature.res_h__amean.r` | `python src/ctd/shared_signal_audit.py` | TeX lines 128 |
| `pdchLatencyp` | 0.086 | `output/shared_signal_audit.json` → `E4_secondary_targets.F2.pdch.per_feature.res_h__amean.p_perm` | `python src/ctd/shared_signal_audit.py` | audit only |
| `pdchLatencyq` | 0.431 | `output/shared_signal_audit.json` → `E4_secondary_targets.F2.pdch.per_feature.res_h__amean.q_bh` | `python src/ctd/shared_signal_audit.py` | TeX lines 128 |
| `binaryHits` | 0 | `output/shared_signal_audit.json` → `E2_concordance.concordance.counts.confirmed_shared` | `python src/ctd/shared_signal_audit.py` | TeX lines 128 |
| `signAgree` | 12 | `output/shared_signal_audit.json` → `E2_concordance.concordance.sign_agreement.n_agree` | `python src/ctd/shared_signal_audit.py` | TeX lines 128 |
| `liveDim` | 22 | `output/shared_signal_audit.json` → `E2_concordance.concordance.sign_agreement.n_comparable` | `python src/ctd/shared_signal_audit.py` | TeX lines 40, 45, 128 |
| `movingLatencyr` | 0.107 | `output/shared_signal_audit.json` → `E4_secondary_targets.F3.daic_phq8_moving.per_feature.res_h__amean.r` | `python src/ctd/shared_signal_audit.py` | TeX lines 130 |
| `movingLatencyq` | 0.520 | `output/shared_signal_audit.json` → `E4_secondary_targets.F3.daic_phq8_moving.per_feature.res_h__amean.q_bh` | `python src/ctd/shared_signal_audit.py` | TeX lines 130 |
| `movingmax_abs_r_observed` | 0.195 | `output/shared_signal_audit.json` → `E4_secondary_targets.F3.daic_phq8_moving.max_abs_r_observed` | `python src/ctd/shared_signal_audit.py` | audit only |
| `movingmax_abs_r_permutation_p` | 0.190 | `output/shared_signal_audit.json` → `E4_secondary_targets.F3.daic_phq8_moving.max_abs_r_permutation_p` | `python src/ctd/shared_signal_audit.py` | audit only |
| `maxBinaryP` | 0.340 | `output/shared_signal_audit.json` → `E2_concordance.pdch.max_abs_r_permutation_p` | `python src/ctd/shared_signal_audit.py` | TeX lines 128 |
| `fdr` | 0.10 | `output/shared_signal_audit.json` → `settings.fdr_q` | `python src/ctd/shared_signal_audit.py` | TeX lines 61 |
| `permutations` | 5000 | `output/shared_signal_audit.json` → `settings.n_permutations` | `python src/ctd/shared_signal_audit.py` | TeX lines 61 |
| `silenceSeconds` | 0.2 | `output/shared_signal_audit.json` → `settings.silence_threshold_s` | `python src/ctd/shared_signal_audit.py` | TeX lines 43 |
| `rankLive` | 13 | `output/shared_signal_audit.json` → `E0_inventory.algebraic_redundancy.daic.ask.matrix_rank_22_live_centered` | `python src/ctd/shared_signal_audit.py` | TeX lines 45 |
| `powerN` | 100 | `output/shared_signal_audit.json` → `E1_confounds.e1c_power_calibration.n_subsamples` | `python src/ctd/shared_signal_audit.py` | TeX lines 132 |
| `powerFmean` | 0.605 | `output/shared_signal_audit.json` → `E1_confounds.e1c_power_calibration.macro_f1.mean` | `python src/ctd/shared_signal_audit.py` | TeX lines 132 |
| `powerFsd` | 0.063 | `output/shared_signal_audit.json` → `E1_confounds.e1c_power_calibration.macro_f1.sd` | `python src/ctd/shared_signal_audit.py` | TeX lines 132 |
| `powerAmean` | 0.636 | `output/shared_signal_audit.json` → `E1_confounds.e1c_power_calibration.roc_auc.mean` | `python src/ctd/shared_signal_audit.py` | TeX lines 132 |
| `powerAsd` | 0.075 | `output/shared_signal_audit.json` → `E1_confounds.e1c_power_calibration.roc_auc.sd` | `python src/ctd/shared_signal_audit.py` | TeX lines 132 |
| `powerFpercentile` | 10 | `output/shared_signal_audit.json` → `E1_confounds.e1c_power_calibration.pdch_located_in_daic_null.macro_f1.percentile_within_daic_at_n62` | `python src/ctd/shared_signal_audit.py` | TeX lines 132 |
| `officialtrainN` | 107 | `output/official_split_ctd.json` → `cohort.train.n` | `python scripts/official_split_sensitivity.py` | TeX lines 124 |
| `officialdevN` | 35 | `output/official_split_ctd.json` → `cohort.dev.n` | `python scripts/official_split_sensitivity.py` | TeX lines 124 |
| `officialtestN` | 47 | `output/official_split_ctd.json` → `cohort.test.n` | `python scripts/official_split_sensitivity.py` | TeX lines 124 |
| `officialN` | 189 | `output/official_split_ctd.json` → `sum(cohort.*.n)` | `python scripts/official_split_sensitivity.py` | TeX lines 38 |
| `excludedN` | 9 | `output/official_split_ctd.json` → `sum(cohort.*.integrity_exclusions_restored)` | `python scripts/official_split_sensitivity.py` | TeX lines 38, 124 |
| `noPairsN` | 3 | `output/official_split_ctd.json` → `sum(length(sessions_without_turn_pairs.*))` | `python scripts/official_split_sensitivity.py` | TeX lines 124 |
| `officialC` | 1.0 | `output/official_split_ctd.json` → `result.selected_C` | `python scripts/official_split_sensitivity.py` | TeX lines 124 |
| `officialDev` | 0.782 | `output/official_split_ctd.json` → `result.dev.macro_f1` | `python scripts/official_split_sensitivity.py` | TeX lines 124 |
| `officialTest` | 0.639 | `output/official_split_ctd.json` → `result.test.macro_f1.point` | `python scripts/official_split_sensitivity.py` | TeX lines 124 |
| `officialAuc` | 0.654 | `output/official_split_ctd.json` → `result.test.roc_auc.point` | `python scripts/official_split_sensitivity.py` | TeX lines 124 |
| `agarwalTest` | 0.80 | `output/revision_review_context.json` → `SLT06.test_macro_f1` | `python scripts/revision_context.py` | TeX lines 34 |
| `wavlmSeedmean` | 0.506 | `output/revision_review_context.json` → `SLT08.wavlm_test_seed_mean` | `python scripts/revision_context.py` | TeX lines 117 |
| `wavlmSeedsd` | 0.034 | `output/revision_review_context.json` → `SLT08.wavlm_test_seed_sd` | `python scripts/revision_context.py` | TeX lines 117 |
| `wavlmSeeds` | 6 | `output/revision_review_context.json` → `SLT08.wavlm_n_seeds` | `python scripts/revision_context.py` | TeX lines 54, 117 |
| `robertaSeedmean` | 0.604 | `output/revision_review_context.json` → `SLT08.roberta_test_seed_mean` | `python scripts/revision_context.py` | TeX lines 99 |
| `robertaSeedsd` | 0.025 | `output/revision_review_context.json` → `SLT08.roberta_test_seed_sd` | `python scripts/revision_context.py` | TeX lines 99 |
| `robertaSeeds` | 3 | `output/revision_review_context.json` → `SLT08.roberta_n_seeds` | `python scripts/revision_context.py` | TeX lines 99 |
| `wavlmSeed` | 44 | `output/revision_review_context.json` → `SLT08.wavlm_selected_seed` | `python scripts/revision_context.py` | TeX lines 54, 115 |
| `outerFolds` | 5 | `src/ctd/pdch_experiment.py` → `N_SPLITS_OUTER` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` | TeX lines 49, 59 |
| `innerFolds` | 4 | `src/ctd/pdch_experiment.py` → `N_SPLITS_INNER` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` | TeX lines 49, 59 |
| `bootstrapN` | 2000 | `src/ctd/ml_splits.py` → `N_BOOTSTRAPS` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` | TeX lines 51 |
| `confidence` | 95 | `src/ctd/ml_splits.py` → `100-ALPHA` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` | TeX lines 51, 103 |
| `threshold` | 0.5 | `src/ctd/ml_splits.py` → `LogisticRegression.predict decision rule` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` | TeX lines 49 |
| `phqCutoff` | 10 | `src/common/daic_cleaning.py` → `PHQ8_DEPRESSED_THRESHOLD` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` | TeX lines 38 |
| `correctedSession` | 409 | `src/common/daic_cleaning.py` → `KNOWN_ERRORS.relabel` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` | TeX lines 38, 124 |
| `hamdCutoff` | 17 | `src/ctd/pdch_labels.py` → `primary label cutoff` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` | TeX lines 40 |
| `annotationSeconds` | 1 | `src/ctd/pdch_adapter.py` → `parse_turns MM:SS annotation resolution` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` | TeX lines 40 |
| `cGrid` | 0.01, 0.03, 0.1, 0.3, 1.0 | `src/ctd/pdch_experiment.py` → `C_GRID` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` | TeX lines 49 |
| `weightStep` | 0.1 | `src/fusion/run_fusion_wavlm_seeds.py` → `weight grid` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` | TeX lines 56 |
| `thresholdLow` | 0.05 | `src/fusion/run_fusion_wavlm_seeds.py` → `threshold grid start` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` | TeX lines 56 |
| `thresholdHigh` | 0.95 | `src/fusion/run_fusion_wavlm_seeds.py` → `threshold grid stop` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` | TeX lines 56 |
| `thresholdStep` | 0.05 | `src/fusion/run_fusion_wavlm_seeds.py` → `threshold grid step` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` | TeX lines 56 |
| `unitSum` | 1 | `output/shared_signal_audit.json` → `E0_inventory.algebraic_redundancy.daic.ask.identity` | `python src/ctd/shared_signal_audit.py` | TeX lines 45 |
| `crossDim` | 7 | `src/ctd/feature_groups.py` → `len(CROSS)` | `python scripts/build_paper_numbers.py; fixed protocol/code definition` | TeX lines 45 |
