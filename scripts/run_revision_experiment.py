#!/usr/bin/env python3
"""Run one revision command and record its environment, source/input hashes and log."""
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "output/revision_20260924"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def snapshot():
    paths = list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.py"))
    paths += list((ROOT / "src/ctd/outputs").glob("functionals_*.csv"))
    paths += list((ROOT / "output/pdch").glob("*.csv"))
    paths += list((ROOT / "output").glob("*.json"))
    data = Path(os.environ.get("DAIC_WOZ_ROOT", "/home/exouser/data/DAIC"))
    paths += list((data / "labels").glob("*.csv"))
    paths += list(data.rglob("*_TRANSCRIPT.csv"))
    paths += list(Path("/home/exouser/data").rglob("*.xlsx"))
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): digest(p)
            for p in sorted(set(paths)) if p.is_file()}

def main():
    tag, *command = sys.argv[1:]
    AUDIT.mkdir(exist_ok=True)
    rec = {"command": command, "cwd": str(ROOT),
           "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
           "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "environment": {k: os.environ.get(k) for k in ("DAIC_WOZ_ROOT", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MPLBACKEND")},
           "python": sys.version,
           "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
           "input_sha256": snapshot()}
    start = time.monotonic()
    with (AUDIT / (tag + ".txt")).open("w") as log:
        status = subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    rec.update(exit_code=status, elapsed_seconds=time.monotonic()-start,
               finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               output_sha256={str(p.relative_to(ROOT)): digest(p) for p in (ROOT / "output").glob("*.json")})
    gate = ROOT / "src/ctd/outputs/ml_splits_results.json"
    if gate.exists():
        rec["gate_md5"] = hashlib.md5(gate.read_bytes()).hexdigest()
        rec["gate_result"] = json.loads(gate.read_text())
    (AUDIT / (tag + ".json")).write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps({k: rec[k] for k in ("command", "exit_code", "elapsed_seconds")}), flush=True)
    sys.exit(status)

if __name__ == "__main__":
    main()
