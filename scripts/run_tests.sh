#!/usr/bin/env bash
# CPU gate: no GPU / pytest required (pytest also works: `pytest tests`).
set -e
cd "$(dirname "$0")/.."
PYTHONPATH=. python3 - <<'PY'
import importlib, time, traceback
mods = ["tests.test_model", "tests.test_data", "tests.test_train"]
n = fails = 0
for m in mods:
    mod = importlib.import_module(m)
    for name in sorted(k for k in dir(mod) if k.startswith("test_")):
        n += 1
        t = time.time()
        try:
            getattr(mod, name)()
            print(f"PASS {m}.{name} ({time.time() - t:.1f}s)")
        except Exception:
            fails += 1
            print(f"FAIL {m}.{name}")
            traceback.print_exc()
print(f"{n - fails}/{n} passed")
raise SystemExit(1 if fails else 0)
PY
PYTHONPATH=. python3 -m morph.eval.kv_bench --config configs/model/s.json
