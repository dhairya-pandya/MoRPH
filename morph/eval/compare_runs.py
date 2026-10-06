"""Summarize runs from their metrics.jsonl: final val CE and self-modeling diagnostics.

    PYTHONPATH=. python -m morph.eval.compare_runs kaggle_out/*/runs/proxy_*
"""
from __future__ import annotations

import argparse
import json
import os


def last_eval(run_dir: str) -> dict:
    recs = [json.loads(ln) for ln in open(os.path.join(run_dir, "metrics.jsonl")) if '"eval"' in ln]
    return recs[-1] if recs else {}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--baseline", default=None, help="run name whose val_ce is the reference")
    args = ap.parse_args(argv)
    rows = {os.path.basename(r.rstrip("/")): last_eval(r) for r in args.runs}
    base = rows.get(args.baseline, {}).get("val_ce") if args.baseline else None
    keys = sorted({k for r in rows.values() for k in r if k.startswith(("erank", "energy", "sm_r2", "wstd/all"))})
    print("run".ljust(16), "tokens".rjust(8), "val_ce".rjust(8), "d%".rjust(7), *[k.rjust(14) for k in keys])
    for name, r in sorted(rows.items()):
        if not r:
            print(name.ljust(16), "no eval yet")
            continue
        d = f"{100 * (r['val_ce'] / base - 1):+.2f}" if base else ""
        print(name.ljust(16), f"{r['tokens'] / 1e9:.3f}B".rjust(8), f"{r['val_ce']:.4f}".rjust(8), d.rjust(7),
              *[f"{r.get(k, float('nan')):.4f}".rjust(14) for k in keys])


if __name__ == "__main__":
    main()
