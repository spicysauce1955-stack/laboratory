"""Diagnostic: what does a GPU box actually offer the workload?

Prints the NVIDIA driver version, the CUDA runtime torch was built against, and whether torch can
see the device. Exists because a GCP GPU job can provision, bill, install the whole CUDA wheel
stack, and still report ``torch.cuda.is_available() == False`` when the image's driver is older
than the CUDA runtime the wheels resolved to.

Writes a JSON summary to $LAB_RUN_DIR and exits 0 regardless — this is a probe, not a gate, so the
answer is recorded even when the accelerator is unusable.

    lab submit -c "python experiments/gpu_probe.py" --backend skypilot --cloud gcp \
        --accelerators L4:1 --with torch --timeout 10m
"""
import json
import os
import subprocess
import sys

info = {}

try:
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=60,
    )
    info["nvidia_smi_rc"] = smi.returncode
    info["nvidia_smi"] = (smi.stdout or smi.stderr).strip()
except Exception as e:  # noqa: BLE001 — the whole point is to record what went wrong
    info["nvidia_smi_rc"] = -1
    info["nvidia_smi"] = f"{type(e).__name__}: {e}"

try:
    import torch

    info["torch_version"] = torch.__version__
    info["torch_cuda_runtime"] = torch.version.cuda
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["device_count"] = torch.cuda.device_count() if info["cuda_available"] else 0
    if not info["cuda_available"]:
        # torch keeps the initialisation failure here; it is the actual diagnosis.
        try:
            import torch.cuda as tc

            info["cuda_init_error"] = str(tc._lazy_init.__doc__ and "" or "")
        except Exception:  # noqa: BLE001
            pass
        try:
            torch.cuda.init()
        except Exception as e:  # noqa: BLE001
            info["cuda_init_error"] = f"{type(e).__name__}: {e}"
except Exception as e:  # noqa: BLE001
    info["torch_error"] = f"{type(e).__name__}: {e}"

info["python"] = sys.version.split()[0]

run_dir = os.environ.get("LAB_RUN_DIR", ".")
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "gpu_probe.json"), "w") as fh:
    json.dump(info, fh, indent=2)

for k, v in info.items():
    print(f"{k}: {v}")
