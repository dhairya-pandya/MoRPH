#!/usr/bin/env bash
# Stage-0 gate: run on CPU, no GPU/pytest required.
set -e
cd "$(dirname "$0")/.."
PYTHONPATH=. python3 -c "
import tests.test_model as T
fns=[getattr(T,n) for n in dir(T) if n.startswith('test_')]
for f in fns: f(); print('PASS', f.__name__)
print(f'{len(fns)}/{len(fns)} passed')
"
PYTHONPATH=. python3 -m morph.eval.kv_bench --config configs/300m_hybrid.json
