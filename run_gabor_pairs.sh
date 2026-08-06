#!/usr/bin/env bash
# Gabor matched-K_eff Q-sweep: one job per (K_eff, Q) (omega0,sigma) pair (calibrated in-memory).
# Tests band-pass>low-pass at matched K_eff_rice + approach to the radial 4.6. Q via kappa_S 2.94->1.12.
set -euo pipefail
cd "$(dirname "$0")"
EXE="uv run --with torch python experiments/v14_kernel_capacity.py"
SEEDS="seeds=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
LEARN="mode=minibatch optimizer=adam lr=0.1 lr_schedule=cosine batch_size=16 n_restarts=8 epochs=5000 patience=5000"
OOM="s_budget=8e9 elem_budget=16e6"
GPU="--backend skypilot --accelerators RTX4090:1 --timeout 45m"
ALPHAS="alphas=2.4,2.8,3.2,3.6,4.0,4.4"

# (Keff Q omega0 sigma)
PAIRS="
32 0.5 0.0401 12.462
32 1.0 0.0577 17.341
32 2.0 0.0605 33.075
32 4.0 0.0634 63.084
64 0.5 0.0800 6.246
64 1.0 0.1151 8.691
64 2.0 0.1225 16.330
64 4.0 0.1265 31.618
"
echo "$PAIRS" | while read -r KE Q OM SG; do
  [ -z "${KE:-}" ] && continue
  echo ">>> gabor Keff=$KE Q=$Q omega0=$OM sigma=$SG"
  uv run lab submit -c "$EXE kernel=gabor omega0=$OM sigma_list=$SG N_list=500 $ALPHAS mean_spikes_fixed=3 $OOM $LEARN $SEEDS" \
    $GPU 2>&1 | grep -E 'job_id'
done
