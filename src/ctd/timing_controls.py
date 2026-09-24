#!/usr/bin/env python3
"""Exploratory nested timing controls; see docs/timing-controls-protocol.md.

Only train/dev are loaded. The official test cohort is never read here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from constants import DEFAULT_DATA_ROOT, OUTPUT_DIR
from feature_groups import FEATURE_CONFIGS
from ml_splits import load_transcript, transcript_path
from pdch_experiment import nested_cv_classification

REPO = Path(__file__).resolve().parents[2]


def run() -> dict:
    paths = [OUTPUT_DIR / f"functionals_{s}.csv" for s in ("train", "dev")]
    data = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    assert len(data) == 135 and data.session_id.nunique() == 135
    assert set(data.split) == {"train", "dev"}
    spans = []
    source_hashes = {}
    for sid in data.session_id:
        path = transcript_path(DEFAULT_DATA_ROOT, int(sid))
        transcript = load_transcript(path)
        spans.append(float(transcript.stop_time.max() - transcript.start_time.min()))
        source_hashes[str(sid)] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert np.all(np.asarray(spans) > 0)
    structure = np.column_stack([np.log(spans), np.log1p(data.n_turns)])
    latency = data[["res_h__amean"]].to_numpy(float)
    designs = {
        name: data[[f"{f}__amean" for f in FEATURE_CONFIGS[name]]].to_numpy(float)
        for name in ("all24", "no_ask")
    }
    designs.update(latency=latency, structure=structure,
                   latency_structure=np.column_stack([latency, structure]))
    results = {}
    for name, x in designs.items():
        print(f"Evaluating {name}: {x.shape[1]} features", flush=True)
        results[name] = nested_cv_classification(
            {0.2: x}, data.PHQ8_Binary.to_numpy(int),
            data.session_id.to_numpy(int), n_repeats=10, fixed_threshold=0.2,
        )
    for path in paths:
        source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "protocol": "docs/timing-controls-protocol.md",
        "n_subjects": len(data), "n_positive": int(data.PHQ8_Binary.sum()),
        "official_test_used": False,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                     "sklearn": sklearn.__version__},
        "source_sha256": source_hashes,
        "results": results,
    }


def main() -> None:
    payload = run()
    out = REPO / "output"
    out.mkdir(exist_ok=True)
    (out / "timing_controls.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# Exploratory timing controls (DAIC train+dev only)", "",
             "135 subjects; 5 outer × 4 inner folds, 10 paired repeats. "
             "Mean ± SD across repeats; SD is not a confidence interval. "
             "No official test data used. See `docs/timing-controls-protocol.md`.", "",
             "| Configuration | Macro-F1 | ROC-AUC |",
             "|---|---:|---:|"]
    for name, result in payload["results"].items():
        f1, auc = result["macro_f1"], result["roc_auc"]
        lines.append(f"| {name} | {f1['mean']:.3f} ± {f1['sd']:.3f} | "
                     f"{auc['mean']:.3f} ± {auc['sd']:.3f} |")
    report = "\n".join(lines) + "\n"
    (out / "timing_controls.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
