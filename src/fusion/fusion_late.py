#!/usr/bin/env python3
"""Late fusion of the ACTUAL deployed single models (reproduces the canonical table).

Unlike train_fusion_v2.py (which re-fits fresh LogReg probes on the penultimate
embeddings), this script fuses each modality's *deployed* probabilities -- the
exact models that produced the project's headline dev/test numbers:

  WavLM-large : sigmoid(classifier . e_a)  using checkpoints/linear_probe/best.pt
                (e_a read from the cached fusion embeddings). threshold 0.5.
  RoBERTa-large: per-session probabilities saved by the deployed seed43 frozen run
                (results/runs/roberta-large_frozen_seed43/{dev,test}_predictions).
                own threshold 0.5238.
  CTD         : canonical 24-D session-mean + L2 LogReg (C=0.3, class_weight
                balanced). dev probs from a train-only fit; test probs from the
                train+dev refit (matches ml_splits.py). threshold 0.5.

So the single-modality rows here REPRODUCE the canonical table:
  WavLM 0.683/0.525   RoBERTa 0.690/0.631   CTD 0.746/0.631   (dev/test macro-F1)

Fusion = combine these probabilities. Per protocol every choice (fusion weights,
threshold) is made on DEV; dev is the primary report, test is on the side; all
with 2000-bootstrap 95% CIs.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from confidence_intervals import evaluate_with_conf_int
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SRC_ROOT = Path(__file__).resolve().parents[1]
CTD_ROOT = SRC_ROOT / "ctd"
for _p in (str(CTD_ROOT), str(SRC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from constants import OUTPUT_DIR  # noqa: E402

FUS = OUTPUT_DIR / "fusion"
WAVLM_CKPT = SRC_ROOT / "acoustic-depr-wavlm" / "checkpoints" / "linear_probe" / "best.pt"
ROB_RUN = (SRC_ROOT / "semantic-depr-roberta" / "results" / "runs"
           / "roberta-large_frozen_seed43")
SPLITS = ("dev", "test")
MODALITIES = ("wavlm", "roberta", "ctd")
OWN_THR = {"wavlm": 0.5, "roberta": 0.5238006711006165, "ctd": 0.5}
N_BOOTSTRAPS = 2000
ALPHA = 5


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def macro_f1(y, p, t):
    from sklearn.metrics import f1_score
    return f1_score(y, (p >= t).astype(int), average="macro")


def best_threshold(y, p):
    ts = np.linspace(0.05, 0.95, 19)
    return float(ts[int(np.argmax([macro_f1(y, p, t) for t in ts]))])


def ci(samples, fn, y):
    point, (lo, hi) = evaluate_with_conf_int(
        np.asarray(samples), fn, np.asarray(y), conditions=None,
        num_bootstraps=N_BOOTSTRAPS, alpha=ALPHA)
    return {"point": float(point), "ci_low": float(lo), "ci_high": float(hi)}


def metric_report(y, probs, thr):
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
    preds = (probs >= thr).astype(int)
    return {
        "macro_f1": ci(preds, lambda yy, s: f1_score(yy, s, average="macro"), y),
        "balanced_accuracy": ci(preds, balanced_accuracy_score, y),
        "f1_depressed": ci(preds, lambda yy, s: f1_score(yy, s, pos_label=1, zero_division=0), y),
        "roc_auc": ci(probs, roc_auc_score, y),
    }


# --------------------------- per-modality deployed probs ----------------------
def wavlm_probs():
    ck = torch.load(WAVLM_CKPT, map_location="cpu", weights_only=False)
    W = ck["model_state"]["classifier.weight"].numpy().reshape(-1)   # [1024]
    b = float(ck["model_state"]["classifier.bias"].numpy().reshape(-1)[0])
    out, lab = {}, {}
    for s in SPLITS:
        d = np.load(FUS / f"wavlm_{s}.npz")
        prob = sigmoid(d["emb"] @ W + b)
        out[s] = {int(i): float(p) for i, p in zip(d["session_id"], prob)}
        lab[s] = {int(i): int(l) for i, l in zip(d["session_id"], d["label"])}
    return out, lab


def roberta_probs():
    out, lab = {}, {}
    files = {"dev": "dev_predictions_best.csv", "test": "test_predictions.csv"}
    for s in SPLITS:
        df = pd.read_csv(ROB_RUN / files[s])
        out[s] = {int(r.participant_id): float(r.probability) for r in df.itertuples()}
        lab[s] = {int(r.participant_id): int(r.label) for r in df.itertuples()}
    return out, lab


def ctd_probs():
    """Canonical 24-D session-mean + L2 LogReg(C=0.3, balanced).

    dev probs: train-only fit (ml_splits dev convention).
    test probs: train+dev refit (ml_splits test convention).
    """
    tr = np.load(FUS / "ctd_train.npz"); dv = np.load(FUS / "ctd_dev.npz"); te = np.load(FUS / "ctd_test.npz")
    Xtr, ytr = tr["emb"], tr["label"]
    Xdv, ydv = dv["emb"], dv["label"]
    Xte = te["emb"]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000).fit(sc.transform(Xtr), ytr)
    dev_p = clf.predict_proba(sc.transform(Xdv))[:, 1]
    # test from train+dev refit
    Xtrd = np.vstack([Xtr, Xdv]); ytrd = np.concatenate([ytr, ydv])
    sc2 = StandardScaler().fit(Xtrd)
    clf2 = LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000).fit(sc2.transform(Xtrd), ytrd)
    test_p = clf2.predict_proba(sc2.transform(Xte))[:, 1]
    out = {"dev": {int(i): float(p) for i, p in zip(dv["session_id"], dev_p)},
           "test": {int(i): float(p) for i, p in zip(te["session_id"], test_p)}}
    lab = {"dev": {int(i): int(l) for i, l in zip(dv["session_id"], dv["label"])},
           "test": {int(i): int(l) for i, l in zip(te["session_id"], te["label"])}}
    return out, lab


def main():
    P = {}   # modality -> split -> {sid: prob}
    L = {}   # modality -> split -> {sid: label}
    P["wavlm"], L["wavlm"] = wavlm_probs()
    P["roberta"], L["roberta"] = roberta_probs()
    P["ctd"], L["ctd"] = ctd_probs()

    # sanity: WavLM reconstruction vs deployed json
    js = json.loads((DEPRESSION_ROOT / "acoustic-depr-wavlm" / "results_test_linear_probe.json").read_text())
    mx = max(abs(P["wavlm"]["test"][r["pid"]] - r["prob"]) for r in js["per_subject"])
    print(f"[sanity] WavLM reconstructed test prob max abs diff vs deployed json: {mx:.2e}")

    # aligned matrices per split
    aligned = {}
    for s in SPLITS:
        common = sorted(set.intersection(*[set(P[m][s]) for m in MODALITIES]))
        y = np.array([L["ctd"][s][i] for i in common], dtype=int)
        probs = {m: np.array([P[m][s][i] for i in common]) for m in MODALITIES}
        aligned[s] = (common, y, probs)
        print(f"  {s}: n={len(common)} depressed={int(y.sum())}")

    ydv = aligned["dev"][1]; yte = aligned["test"][1]
    Pdv = aligned["dev"][2]; Pte = aligned["test"][2]

    results = {}

    def record(key, dev_p, test_p, thr, extra=None):
        dev_rep = metric_report(ydv, dev_p, thr)
        test_rep = metric_report(yte, test_p, thr)
        results[key] = {"threshold": float(thr), "dev": dev_rep, "test": test_rep}
        if extra:
            results[key].update(extra)
        dm, tm = dev_rep["macro_f1"], test_rep["macro_f1"]
        print(f"  {key:<30} dev={dm['point']:.3f} [{dm['ci_low']:.3f},{dm['ci_high']:.3f}]"
              f"  | test={tm['point']:.3f} [{tm['ci_low']:.3f},{tm['ci_high']:.3f}]")
        return dev_rep["macro_f1"]["point"]

    # ---- single modalities at their OWN deployed thresholds (reproduce table) ----
    print("\n--- Single modalities (deployed models; own thresholds) ---")
    for m in MODALITIES:
        record(f"single_{m}", Pdv[m], Pte[m], OWN_THR[m])

    # ---- fusion: equal-weight (parameter-free); threshold tuned on dev ----
    print("\n--- Fusion: equal-weight mean_prob (threshold tuned on dev) ---")
    subsets = [list(c) for k in (2, 3) for c in combinations(MODALITIES, k)]
    for sub in subsets:
        dv = np.mean([Pdv[m] for m in sub], 0)
        te = np.mean([Pte[m] for m in sub], 0)
        record(f"mean_prob[{'+'.join(sub)}]", dv, te, best_threshold(ydv, dv))

    print("\n--- Fusion: mean_logit (threshold tuned on dev) ---")
    for sub in subsets:
        dv = sigmoid(np.mean([logit(Pdv[m]) for m in sub], 0))
        te = sigmoid(np.mean([logit(Pte[m]) for m in sub], 0))
        record(f"mean_logit[{'+'.join(sub)}]", dv, te, best_threshold(ydv, dv))

    # ---- fusion: convex weights + threshold both grid-searched on DEV ----
    print("\n--- Fusion: convex weights (grid-searched on dev) ---")
    grid = np.arange(0.0, 1.0001, 0.1)
    for sub in subsets:
        k = len(sub)
        if k == 2:
            combos = [np.array([a, 1 - a]) for a in grid]
        else:
            combos = [np.array([a, b, 1 - a - b]) for a in grid for b in grid if a + b <= 1.0 + 1e-9]
        Pd = np.stack([Pdv[m] for m in sub], 1)
        Pt = np.stack([Pte[m] for m in sub], 1)
        best = None
        for w in combos:
            dvp = Pd @ w
            t = best_threshold(ydv, dvp)
            f = macro_f1(ydv, dvp, t)
            if best is None or f > best[0]:
                best = (f, w, t)
        _, w, t = best
        record(f"wconvex[{'+'.join(sub)}]", Pd @ w, Pt @ w, t,
               {"weights": {m: float(round(wi, 3)) for m, wi in zip(sub, w)}})

    best_key = max(results, key=lambda k: results[k]["dev"]["macro_f1"]["point"])
    bd = results[best_key]["dev"]["macro_f1"]; bt = results[best_key]["test"]["macro_f1"]
    print(f"\nDev-selected best: {best_key}")
    print(f"  dev  {bd['point']:.3f} [{bd['ci_low']:.3f},{bd['ci_high']:.3f}] (primary)")
    print(f"  test {bt['point']:.3f} [{bt['ci_low']:.3f},{bt['ci_high']:.3f}] (side)")

    out = OUTPUT_DIR / "fusion_late_results.json"
    out.write_text(json.dumps({
        "description": "late fusion of DEPLOYED single models; singles reproduce the canonical table",
        "n_dev": len(aligned["dev"][0]), "n_test": len(aligned["test"][0]),
        "own_thresholds": OWN_THR, "dev_selected_best": best_key, "results": results,
    }, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
