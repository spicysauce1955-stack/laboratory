#!/usr/bin/env bash
# LEAN gap-targeted re-run: corrected gabor (inversion, not width-override), higher alpha grids,
# leaner learner (epochs=3500 restarts=3 seeds=12) so cells finish under the 55m cap.
set -euo pipefail
cd "$(dirname "$0")"
EXE="uv run --with torch python experiments/v14_kernel_capacity.py"
SEEDS="seeds=0,1,2,3,4,5,6,7,8,9,10,11"
LEARN="mode=minibatch optimizer=adam lr=0.1 lr_schedule=cosine batch_size=16 n_restarts=3 epochs=3500 patience=3500"
OOM="s_budget=8e9 elem_budget=16e6"
GPU="--backend skypilot --accelerators RTX4090:1 --timeout 55m"

case "${1:-}" in
gauss)   # high-K cells need higher alpha (have K=4/8/16 already)
  uv run lab sweep -c "$EXE kernel=gauss N_list=500 alphas=2.8,3.2,3.6,4.0,4.4 mean_spikes_fixed=3 capture_overlap=1 $OOM $LEARN $SEEDS" \
    --grid Keff_list=32,64,128 $GPU ;;
gabor)   # CORRECTED: Keff-inversion (omega0 grid), matched-Keff band-pass vs low-pass at 32 & 64
  uv run lab sweep -c "$EXE kernel=gabor N_list=500 alphas=2.8,3.4,4.0,4.6 mean_spikes_fixed=3 $OOM $LEARN $SEEDS" \
    --grid omega0=0.02,0.06,0.12 --grid Keff_list=32,64 $GPU ;;
sinusoid) # need higher alpha to find ~3.4 crossing + flatness
  uv run lab sweep -c "$EXE kernel=sinusoid N_list=500 alphas=3.2,3.6,4.0,4.4 mean_spikes_fixed=3 capture_overlap=1 n_restarts=8 mode=minibatch optimizer=adam lr=0.1 lr_schedule=cosine batch_size=16 epochs=3500 patience=3500 $OOM $SEEDS" \
    --grid Keff_list=8,16,32,64 $GPU ;;
alpha)   # I4: alpha vs alpha_trunc at identical tau; need data (all prior failed)
  uv run lab sweep -c "$EXE N_list=500 alphas=2.0,2.6,3.2,3.6 mean_spikes_fixed=3 $OOM $LEARN $SEEDS" \
    --grid kernel=alpha,alpha_trunc --grid width_list=31.25,15.625,7.8125 $GPU ;;
rect)    # smaller-tau cells need higher alpha + width=125/N=600 gap
  uv run lab sweep -c "$EXE kernel=rect alphas=2.4,2.8,3.2,3.6 mean_spikes_fixed=3 $OOM $LEARN $SEEDS" \
    --grid width_list=62.5 --grid N_list=300,600 --grid oversample=4,8 $GPU ;;
*) echo "usage: $0 {gauss|gabor|sinusoid|alpha|rect}"; exit 2 ;;
esac
