#!/usr/bin/env python3
"""Verify all modalities use the SAME dev/test (and train) sessions + labels.

Compares the session_id/label arrays stored in the fusion .npz files
(wavlm/roberta/ctd, per split) against each other and against the canonical
AVEC2017 split CSVs (via ml_splits.load_split_labels, which applies the project
cleaning). Flags any session that is in one modality but not another, any label
disagreement, and any modality whose "test" set actually equals the canonical
dev set (the dev-named-as-test failure mode).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
CTD_ROOT = SRC_ROOT / "ctd"
for _p in (str(CTD_ROOT), str(SRC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from constants import OUTPUT_DIR  # noqa: E402
from ml_splits import load_split_labels  # noqa: E402

FUS = OUTPUT_DIR / "fusion"
SPLITS = ("train", "dev", "test")
MODALITIES = ("wavlm", "roberta", "ctd")


def npz_split(mod, split):
    d = np.load(FUS / f"{mod}_{split}.npz")
    return {int(i): int(l) for i, l in zip(d["session_id"], d["label"])}


def canonical_split(split):
    df = load_split_labels(split)
    return {int(r.session_id): int(r.PHQ8_Binary) for r in df.itertuples()}


def main():
    print("=" * 78)
    print("CANONICAL AVEC2017 SPLITS (ml_splits.load_split_labels, cleaned)")
    canon = {s: canonical_split(s) for s in SPLITS}
    for s in SPLITS:
        ids = sorted(canon[s])
        print(f"  {s:<5} n={len(ids):<3} depressed={sum(canon[s].values()):<3} "
              f"ids[:5]={ids[:5]} ... ids[-3:]={ids[-3:]}")

    # cross-split overlap sanity (must be disjoint)
    print("\nCanonical cross-split overlaps (must all be 0):")
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                ov = set(canon[a]) & set(canon[b])
                print(f"  {a} ∩ {b}: {len(ov)} {sorted(ov) if ov else ''}")

    print("\n" + "=" * 78)
    print("PER-MODALITY .npz vs CANONICAL")
    all_ok = True
    for mod in MODALITIES:
        print(f"\n--- {mod} ---")
        for s in SPLITS:
            m = npz_split(mod, s)
            mids, cids = set(m), set(canon[s])
            missing = cids - mids        # canonical sessions absent from modality
            extra = mids - cids          # modality sessions not in canonical split
            common = mids & cids
            lbl_mismatch = [i for i in common if m[i] != canon[s][i]]
            status = "OK" if not (extra or lbl_mismatch) else "PROBLEM"
            if extra or lbl_mismatch:
                all_ok = False
            print(f"  {s:<5} npz_n={len(m):<3} canon_n={len(cids):<3} "
                  f"common={len(common):<3} missing_from_npz={len(missing):<2} "
                  f"extra_in_npz={len(extra):<2} label_mismatch={len(lbl_mismatch):<2} -> {status}")
            if missing:
                print(f"        (canonical sessions not cached for {mod}: {sorted(missing)})")
            if extra:
                print(f"        !! npz sessions NOT in canonical {s}: {sorted(extra)}")
            if lbl_mismatch:
                print(f"        !! label mismatches: {sorted(lbl_mismatch)}")

            # dev-named-as-test detector: does this modality's split match a DIFFERENT canonical split?
            for other in SPLITS:
                if other != s and set(m) == set(canon[other]):
                    print(f"        !! WARNING: {mod} '{s}' set EQUALS canonical '{other}' set")
                    all_ok = False

    print("\n" + "=" * 78)
    print("CROSS-MODALITY AGREEMENT (wavlm vs roberta vs ctd)")
    for s in SPLITS:
        sets = {mod: set(npz_split(mod, s)) for mod in MODALITIES}
        inter = set.intersection(*sets.values())
        union = set.union(*sets.values())
        print(f"\n  {s}: " + "  ".join(f"{mod}={len(sets[mod])}" for mod in MODALITIES)
              + f"  | intersection={len(inter)} union={len(union)}")
        for mod in MODALITIES:
            only = sets[mod] - inter
            if only:
                print(f"      {mod} has {len(only)} session(s) the others lack: {sorted(only)}")
        # label agreement on the intersection
        lbls = {mod: npz_split(mod, s) for mod in MODALITIES}
        disagree = [i for i in inter
                    if len({lbls[mod][i] for mod in MODALITIES}) > 1]
        if disagree:
            all_ok = False
            print(f"      !! label disagreement on intersection: {sorted(disagree)}")
        else:
            print("      labels agree across modalities on the intersection")

    print("\n" + "=" * 78)
    print("VERDICT:", "ALL CONSISTENT ✓" if all_ok else "INCONSISTENCIES FOUND — see above")


if __name__ == "__main__":
    main()
