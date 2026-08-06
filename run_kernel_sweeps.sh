#!/usr/bin/env bash
# Per-kernel capacity sweeps (v14). Learner validated by pre-flight (Adam-cosine lr=0.1 batch=16,
# graded P_solve + critical slowing). One job per grid cell; 16 seeds internal; incremental CSV.
# Cost: skypilot RTX4090, hard --timeout/cell, fixed density, OOM budgets. Each sweep -> a sweep_id.
set -euo pipefail
cd "$(dirname "$0")"
EXE="uv run --with torch python experiments/v14_kernel_capacity.py"
SEEDS="seeds=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
LEARN="mode=minibatch optimizer=adam lr=0.1 lr_schedule=cosine batch_size=16 n_restarts=8 epochs=5000 patience=5000"
OOM="s_budget=8e9 elem_budget=16e6"
GPU="--backend skypilot --accelerators RTX4090:1 --timeout 45m"

case "${1:-}" in
gauss)
  uv run lab sweep -c "$EXE kernel=gauss N_list=500 alphas=2.0,2.4,2.8,3.2,3.6,4.0 mean_spikes_fixed=3 capture_overlap=1 $OOM $LEARN $SEEDS" \
    --grid Keff_list=4,8,16,32,64,128 $GPU ;;
alpha)
  uv run lab sweep -c "$EXE N_list=500 alphas=2.0,2.4,2.8,3.2,3.6 mean_spikes_fixed=3 $OOM $LEARN $SEEDS" \
    --grid kernel=alpha,alpha_trunc --grid width_list=31.25,15.625,7.8125 $GPU ;;
rect)
  uv run lab sweep -c "$EXE kernel=rect alphas=1.6,2.0,2.4,2.8 mean_spikes_fixed=3 $OOM $LEARN $SEEDS" \
    --grid width_list=62.5,125 --grid N_list=300,600 --grid oversample=4,8 $GPU ;;
sinusoid)
  uv run lab sweep -c "$EXE kernel=sinusoid N_list=500 alphas=2.8,3.2,3.6,4.0,4.2 mean_spikes_fixed=3 capture_overlap=1 n_restarts=16 mode=minibatch optimizer=adam lr=0.1 lr_schedule=cosine batch_size=16 epochs=5000 patience=5000 $OOM $SEEDS" \
    --grid Keff_list=8,16,32,64 $GPU ;;
*) echo "usage: $0 {gauss|alpha|rect|sinusoid}"; exit 2 ;;
esac
