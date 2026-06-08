#!/usr/bin/env bash
# V10 per-epoch learning curves: run a few representative setups at N=200, alpha=2.4 and stream the
# per-epoch training error (log_every=25) so we can plot accuracy(=1-err) vs epoch per setup. Each
# setup is timeout-wrapped (hard cap) so nothing runs away. All output to stdout (durable logs.txt).
set -uo pipefail
cd "$(dirname "$0")/.."
RUN="${LAB_RUN_DIR:-/tmp/v10_curves}"; mkdir -p "$RUN"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
python -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "FATAL: no CUDA device."; exit 2; }
COMMON="ensemble=single vthr_fixed=1.0 K_list=66.667 grid_per_corr=8 N_list=200 alphas=2.4 n_seeds=16 epochs=5000 log_every=25"
run () { echo "=== CURVE setup=$1 ==="; LAB_RUN_DIR="$RUN/$1" timeout --signal=INT 700 \
         python experiments/v3_capacity_sweep.py $COMMON $2 || echo "(setup $1 hit 700s cap or errored)"; }

run adam-b16-cosine    "mode=minibatch batch_size=16 optimizer=adam lr_mode=fixed lr=6.859e-3 adam_b1=0.95 adam_b2=0.999 adam_eps=1e-6 lr_schedule=cosine lr_warmup=309"
run adam-b16-constant  "mode=minibatch batch_size=16 optimizer=adam lr_mode=fixed lr=6.859e-3 adam_b1=0.95 adam_b2=0.999 adam_eps=1e-6 lr_schedule=none lr_warmup=0"
run sgd-b1-cosine      "mode=minibatch batch_size=1 optimizer=momentum momentum=0.99 lr_mode=gs lr_gs_coeff=1.662e-2 lr_schedule=cosine lr_warmup=149"
run sgd-b1-faithful    "mode=online momentum=0.99 lr_mode=gs lr_gs_coeff=3e-3 lr_schedule=none"
run rmsprop-b64-cosine "mode=minibatch batch_size=64 optimizer=rmsprop lr_mode=fixed lr=1.295e-2 rms_alpha=0.99 rms_eps=1e-8 lr_schedule=cosine lr_warmup=128"
run sgd-b16-cosine     "mode=minibatch batch_size=16 optimizer=momentum momentum=0.99 lr_mode=gs lr_gs_coeff=1.662e-2 lr_schedule=cosine lr_warmup=149"
echo "[v10_curves] done"
