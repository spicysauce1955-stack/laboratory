#!/usr/bin/env bash
# V10 confirmation cell-sweep (Sec 11/14.5): run studies/v3_capacity_sweep.py with a winner (or the
# faithful baseline) config, capture raw cells, then run v9_derive ON THE REMOTE so the headline
# alpha_c (1/2-crossing + bootstrap CI + divergence-fit) lands in the durable logs.txt even if the
# .npz rsync misses. Usage: bash experiments/v10_confirm_run.sh "<v3 key=value args>"
set -uo pipefail
cd "$(dirname "$0")/.."
RUN="${LAB_RUN_DIR:-/tmp/v10_confirm}"; mkdir -p "$RUN"
ARGS="${1:?v3 args}"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
if ! python -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "FATAL: no CUDA device; refusing CPU confirmation run."; exit 2
fi
echo "=== v3 sweep (HARD in-instance cap ${SWEEP_TIMEOUT:-5400}s; do NOT trust the lab --timeout): $ARGS ==="
# Wrap the sweep in `timeout` so the COMPUTE is hard-killed on the instance itself, independent of the
# lab supervisor (whose --timeout has twice failed to enforce, $136 runaway). On a kill we still derive
# from whatever cells completed, so partial results survive.
LAB_RUN_DIR="$RUN" timeout --signal=INT "${SWEEP_TIMEOUT:-5400}" \
   python experiments/v3_capacity_sweep.py $ARGS || echo "(sweep hit the ${SWEEP_TIMEOUT:-5400}s cap or errored; deriving on partial cells)"
echo "=== v9_derive (headline alpha_c -> durable log) ==="
python experiments/v9_derive.py "$RUN" --out "$RUN/derived.json" || echo "(derive failed; cells in $RUN/cells)"
echo "=== DERIVED_JSON ==="; cat "$RUN/derived.json" 2>/dev/null
echo "=== MANIFEST ==="; cat "$RUN/manifest.json" 2>/dev/null
echo "[v10_confirm] done"
