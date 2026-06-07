#!/usr/bin/env bash
# V10 remote DEMO (not the confirmatory study): verify the pipeline + metric collection on a GPU.
# (1) run the v10 minibatch parity/rung/mask/solver gates; (2) run a tiny FTE-budgeted HPO per arm,
# writing trials.jsonl/best_config/manifest per arm under $LAB_RUN_DIR/<arm>/. Cheap + fast by design.
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root (parent of experiments/)
RUN="${LAB_RUN_DIR:-/tmp/v10_demo}"
mkdir -p "$RUN"
echo "=== [demo] python/torch banner ==="
python -c "import torch,optuna,numpy; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'optuna',optuna.__version__)"

echo "=== [demo] pytest: v10 minibatch gates ==="
python -m pytest experiments/test_v10_minibatch.py -q 2>&1 | tee "$RUN/pytest.txt"
PYTEST_RC=${PIPESTATUS[0]}
echo "[demo] pytest rc=$PYTEST_RC"

echo "=== [demo] tiny HPO per arm (tune_N=40 n_seeds=4 R=20->180 eta=3) ==="
for ARM in sgd adam rmsprop random; do
  BUD=3; [ "$ARM" = "random" ] && BUD=6   # random baseline gets 2x compute (Sec 10)
  echo "--- arm=$ARM budget_fte=$BUD ---"
  LAB_RUN_DIR="$RUN/$ARM" python experiments/v10_hpo.py \
     arm=$ARM budget_fte=$BUD tune_N=40 n_seeds=4 alpha_obj=2.3,2.5 \
     R_min=20 R_max=180 eta=3 max_trials=60
done

echo "=== [demo] artifact tree ==="
find "$RUN" -maxdepth 2 -type f | sort
echo "=== [demo] trials per arm ==="
for ARM in sgd adam rmsprop random; do
  n=$(wc -l < "$RUN/$ARM/trials.jsonl" 2>/dev/null || echo 0)
  echo "  $ARM: $n trial records"
done
echo "[demo] done (pytest rc=$PYTEST_RC)"
exit $PYTEST_RC
