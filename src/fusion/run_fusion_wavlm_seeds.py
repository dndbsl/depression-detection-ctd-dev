#!/usr/bin/env python3
"""Late fusion with a chosen WavLM seed checkpoint + its dev-tuned threshold.

RoBERTa and CTD use the deployed rerun artifacts. WavLM probabilities are
recomputed from cache with the FULL model (layer-weighting + attention + head)
because penultimate embeddings differ per seed.

Usage:
    python run_fusion_wavlm_seeds.py --wavlm-seeds 43 44
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from confidence_intervals import evaluate_with_conf_int
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

SRC_ROOT = Path(__file__).resolve().parents[1]
CTD_ROOT = SRC_ROOT / "ctd"
WAVLM_ROOT = SRC_ROOT / "acoustic-depr-wavlm"
sys.path.insert(0, str(WAVLM_ROOT))

from dataset import SubjectFeatureDataset, collate_fn  # noqa: E402
from model.full_model import build_model  # noqa: E402

ROB_RUN = SRC_ROOT / "semantic-depr-roberta" / "results" / "runs" / "roberta-large_frozen_seed43"
EMB = CTD_ROOT / "outputs" / "fusion"  # written by fusion/extract_ctd_roberta.py
OUT_DIR = CTD_ROOT / "outputs" / "fusion_by_wavlm_seed"

MODALITIES = ("wavlm", "roberta", "ctd")
SPLITS = ("dev", "test")
ROBERTA_THR = 0.5238006711006165
CTD_THR = 0.5
N_BOOTSTRAPS = 2000
ALPHA = 5


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def macro_f1(y, p, t):
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
    preds = (probs >= thr).astype(int)
    return {
        "macro_f1": ci(preds, lambda yy, s: f1_score(yy, s, average="macro"), y),
        "balanced_accuracy": ci(preds, balanced_accuracy_score, y),
        "f1_depressed": ci(preds, lambda yy, s: f1_score(yy, s, pos_label=1, zero_division=0), y),
        "roc_auc": ci(probs, roc_auc_score, y),
    }


@torch.no_grad()
def wavlm_probs_for_seed(seed: int, device: str = "cpu"):
    ckpt_path = WAVLM_ROOT / "checkpoints" / f"seed_{seed}" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing WavLM checkpoint: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ck["config"]).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    wavlm_thr = float(ck.get("threshold", 0.5))

    from torch.utils.data import DataLoader

    out, lab = {}, {}
    for split in SPLITS:
        ds = SubjectFeatureDataset(split, str(WAVLM_ROOT / "manifests"), str(WAVLM_ROOT / "cache"))
        loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
        sids, probs, labels = [], [], []
        for batch in loader:
            feats = batch["features"].to(device)
            mask = batch["mask"].to(device)
            logits, _ = model(feats, mask)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(batch["labels"].numpy())
            sids.extend(int(p) for p in batch["pids"])
        out[split] = {sid: float(p) for sid, p in zip(sids, np.concatenate(probs))}
        lab[split] = {sid: int(l) for sid, l in zip(sids, np.concatenate(labels).astype(int))}
    return out, lab, wavlm_thr, ckpt_path, int(ck["epoch"]), float(ck["dev_loss"])


def roberta_probs():
    out, lab = {}, {}
    files = {"dev": "dev_predictions_best.csv", "test": "test_predictions.csv"}
    for s in SPLITS:
        df = pd.read_csv(ROB_RUN / files[s])
        out[s] = {int(r.participant_id): float(r.probability) for r in df.itertuples()}
        lab[s] = {int(r.participant_id): int(r.label) for r in df.itertuples()}
    return out, lab


def ctd_probs():
    tr = np.load(EMB / "ctd_train.npz")
    dv = np.load(EMB / "ctd_dev.npz")
    te = np.load(EMB / "ctd_test.npz")
    Xtr, ytr = tr["emb"], tr["label"]
    Xdv, ydv = dv["emb"], dv["label"]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000).fit(sc.transform(Xtr), ytr)
    dev_p = clf.predict_proba(sc.transform(Xdv))[:, 1]
    Xtrd = np.vstack([Xtr, Xdv])
    ytrd = np.concatenate([ytr, ydv])
    sc2 = StandardScaler().fit(Xtrd)
    clf2 = LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000).fit(sc2.transform(Xtrd), ytrd)
    test_p = clf2.predict_proba(sc2.transform(te["emb"]))[:, 1]
    out = {
        "dev": {int(i): float(p) for i, p in zip(dv["session_id"], dev_p)},
        "test": {int(i): float(p) for i, p in zip(te["session_id"], test_p)},
    }
    lab = {
        "dev": {int(i): int(l) for i, l in zip(dv["session_id"], dv["label"])},
        "test": {int(i): int(l) for i, l in zip(te["session_id"], te["label"])},
    }
    return out, lab


def run_fusion_for_seed(seed: int, device: str = "cpu") -> dict:
    P_w, L_w, wavlm_thr, ckpt_path, epoch, dev_loss = wavlm_probs_for_seed(seed, device)
    P_r, L_r = roberta_probs()
    P_c, L_c = ctd_probs()

    P = {"wavlm": P_w, "roberta": P_r, "ctd": P_c}
    L = {"wavlm": L_w, "roberta": L_r, "ctd": L_c}
    own_thr = {"wavlm": wavlm_thr, "roberta": ROBERTA_THR, "ctd": CTD_THR}

    aligned = {}
    for s in SPLITS:
        common = sorted(set.intersection(*[set(P[m][s]) for m in MODALITIES]))
        y = np.array([L["ctd"][s][i] for i in common], dtype=int)
        probs = {m: np.array([P[m][s][i] for i in common]) for m in MODALITIES}
        aligned[s] = (common, y, probs)

    ydv, yte = aligned["dev"][1], aligned["test"][1]
    Pdv, Pte = aligned["dev"][2], aligned["test"][2]
    results = {}

    def record(key, dvp, tep, thr, extra=None):
        dev_rep = metric_report(ydv, dvp, thr)
        test_rep = metric_report(yte, tep, thr)
        results[key] = {"threshold": float(thr), "dev": dev_rep, "test": test_rep}
        if extra:
            results[key].update(extra)
        dm, tm = dev_rep["macro_f1"], test_rep["macro_f1"]
        print(f"  {key:<34} dev={dm['point']:.3f} [{dm['ci_low']:.3f},{dm['ci_high']:.3f}]"
              f"  | test={tm['point']:.3f} [{tm['ci_low']:.3f},{tm['ci_high']:.3f}]")
        return dev_rep["macro_f1"]["point"]

    print(f"\n{'='*72}")
    print(f"WavLM seed {seed}  epoch={epoch}  dev_loss={dev_loss:.4f}  wavlm_thr={wavlm_thr:.3f}")
    print(f"checkpoint: {ckpt_path}")
    print(f"{'='*72}")

    print("\n--- Single modalities (own thresholds) ---")
    for m in MODALITIES:
        record(f"single_{m}", Pdv[m], Pte[m], own_thr[m])

    subsets = [list(c) for k in (2, 3) for c in combinations(MODALITIES, k)]

    print("\n--- wconvex (weights + threshold on dev) ---")
    grid = np.arange(0.0, 1.0001, 0.1)
    for sub in subsets:
        if len(sub) == 2:
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
    bd = results[best_key]["dev"]["macro_f1"]
    bt = results[best_key]["test"]["macro_f1"]
    print(f"\nDev-selected best fusion: {best_key}")
    print(f"  dev  {bd['point']:.3f} [{bd['ci_low']:.3f},{bd['ci_high']:.3f}]")
    print(f"  test {bt['point']:.3f} [{bt['ci_low']:.3f},{bt['ci_high']:.3f}]")

    payload = {
        "wavlm_seed": seed,
        "wavlm_checkpoint": str(ckpt_path),
        "wavlm_epoch": epoch,
        "wavlm_dev_loss": dev_loss,
        "own_thresholds": own_thr,
        "roberta_run": str(ROB_RUN),
        "n_dev": len(aligned["dev"][0]),
        "n_test": len(aligned["test"][0]),
        "dev_selected_best": best_key,
        "results": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"fusion_wavlm_seed{seed}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved -> {out_path}")
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wavlm-seeds", nargs="+", type=int, default=[43, 44])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    summaries = []
    for seed in args.wavlm_seeds:
        payload = run_fusion_for_seed(seed, device=args.device)
        summaries.append({
            "seed": seed,
            "wavlm_thr": payload["own_thresholds"]["wavlm"],
            "single_wavlm_test": payload["results"]["single_wavlm"]["test"]["macro_f1"]["point"],
            "single_roberta_test": payload["results"]["single_roberta"]["test"]["macro_f1"]["point"],
            "single_ctd_test": payload["results"]["single_ctd"]["test"]["macro_f1"]["point"],
            "best_fusion": payload["dev_selected_best"],
            "best_fusion_dev": payload["results"][payload["dev_selected_best"]]["dev"]["macro_f1"]["point"],
            "best_fusion_test": payload["results"][payload["dev_selected_best"]]["test"]["macro_f1"]["point"],
        })

    print(f"\n{'='*72}")
    print("SUMMARY (test macro-F1)")
    print(f"{'seed':>4} {'w_thr':>6} {'wavlm':>7} {'roberta':>7} {'ctd':>7} | {'best_fusion':>28} {'fus_dev':>7} {'fus_test':>8}")
    for s in summaries:
        print(f"{s['seed']:>4} {s['wavlm_thr']:>6.2f} {s['single_wavlm_test']:>7.3f} {s['single_roberta_test']:>7.3f} "
              f"{s['single_ctd_test']:>7.3f} | {s['best_fusion']:>28} {s['best_fusion_dev']:>7.3f} {s['best_fusion_test']:>8.3f}")

    summary_path = OUT_DIR / "summary_wavlm_seeds.json"
    summary_path.write_text(json.dumps(summaries, indent=2))
    print(f"\nSaved summary -> {summary_path}")


if __name__ == "__main__":
    main()
