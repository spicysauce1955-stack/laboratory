#!/usr/bin/env bash
# V10 remote DEMO #2: exercise the OBJECTIVE dynamic range + Hyperband pruning at SUB-capacity loads
# (alpha={1.0,1.5}), where p_solve spans 0->1 within the budget so TPE/Hyperband have signal. Dumps
# every artifact to stdout at the end so results survive a teardown-rsync miss (logs = source of truth).
set -uo pipefail
cd "$(dirname "$0")/.."
RUN="${LAB_RUN_DIR:-/tmp/v10_demo2}"
mkdir -p "$RUN"
python -c "import torch,optuna; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'optuna',optuna.__version__)"

for ARM in sgd adam; do
  echo "=== arm=$ARM (sub-capacity alpha=1.0,1.5) ==="
  LAB_RUN_DIR="$RUN/$ARM" python experiments/v10_hpo.py \
     arm=$ARM budget_fte=12 tune_N=40 n_seeds=6 alpha_obj=1.0,1.5 \
     R_min=15 R_max=150 eta=3 max_trials=50
done

for ARM in sgd adam; do
  echo "=== TRIALS_JSONL arm=$ARM ==="; cat "$RUN/$ARM/trials.jsonl" 2>/dev/null
  echo "=== BEST_CONFIG arm=$ARM ==="; cat "$RUN/$ARM/best_config.json" 2>/dev/null
  echo "=== MANIFEST arm=$ARM ==="; cat "$RUN/$ARM/manifest.json" 2>/dev/null
done
echo "[demo2] done"
