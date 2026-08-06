#!/usr/bin/env bash
# $70 / 24h deepening push. Higher fidelity (24 seeds, 90m cap, budget allows). v14 gabor fix applied.
set -euo pipefail
cd "$(dirname "$0")"
EXE="uv run --with torch python experiments/v14_kernel_capacity.py"
S24="seeds=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23"
S16="seeds=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
LEARN="mode=minibatch optimizer=adam lr=0.1 lr_schedule=cosine batch_size=16 epochs=4000 patience=4000"
OOM="s_budget=8e9 elem_budget=16e6"
GPU="--backend skypilot --accelerators RTX4090:1 --timeout 90m"

case "${1:-}" in
rect)    # SOMPOLINSKY BENCHMARK: K=T/tau in {2,3,4,8,16} (tau=250/166.667/125/62.5/31.25), N-phase 1000 & 500
  uv run lab sweep -c "$EXE kernel=rect alphas=1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4 mean_spikes_fixed=3 n_restarts=4 $OOM $LEARN $S24" \
    --grid width_list=250,166.667,125,62.5,31.25 --grid N_list=1000,500 $GPU ;;
gauss)   # extend K to separate ln K vs ln ln K (combine with existing K=4..32)
  uv run lab sweep -c "$EXE kernel=gauss N_list=500 alphas=3.0,3.2,3.4,3.6,3.8,4.0,4.2,4.4 mean_spikes_fixed=3 capture_overlap=1 n_restarts=4 $OOM $LEARN $S24" \
    --grid Keff_list=64,128,256,512 $GPU ;;
sinusoid) # heavy restart: is findable 3.4 learner-limited or the RSB floor?
  uv run lab sweep -c "$EXE kernel=sinusoid N_list=500 alphas=3.4,3.8,4.2,4.6 mean_spikes_fixed=3 n_restarts=32 mode=minibatch optimizer=adam lr=0.1 lr_schedule=cosine batch_size=16 epochs=4000 patience=4000 $OOM $S16" \
    --grid Keff_list=16,32 $GPU ;;
*) echo "usage: $0 {rect|gauss|sinusoid}"; exit 2 ;;
esac
