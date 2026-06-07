#!/usr/bin/env bash
# V10 CONFIRMATORY per-arm run (frozen EXPERIMENT-SPEC.md Sec 10/12 config). One arm per lab job.
# Usage: bash experiments/v10_run.sh <arm> <budget_fte>.  Dumps artifacts to stdout at the end so
# results survive a teardown-rsync miss (logs.txt = durable source of truth).
set -uo pipefail
cd "$(dirname "$0")/.."
ARM="${1:?arm}"; BUD="${2:?budget_fte}"
RUN="${LAB_RUN_DIR:-/tmp/v10_$ARM}"; mkdir -p "$RUN"
python -c "import torch,optuna;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'optuna',optuna.__version__)"
LAB_RUN_DIR="$RUN" python experiments/v10_hpo.py \
   arm="$ARM" budget_fte="$BUD" tune_N=200 n_seeds=16 alpha_obj=2.3,2.5 \
   R_min=200 R_max=5000 eta=3
echo "=== TRIALS_JSONL arm=$ARM ==="; cat "$RUN/trials.jsonl" 2>/dev/null
echo "=== BEST_CONFIG arm=$ARM ==="; cat "$RUN/best_config.json" 2>/dev/null
echo "=== MANIFEST arm=$ARM ==="; cat "$RUN/manifest.json" 2>/dev/null
echo "[v10_run] done arm=$ARM"
