"""Audit probe runner: execute the four recovered audit probes, print all stdout.

Read-only numerical checks written by the research-audit reviewers; no project
imports, no training. Each probe is executed in its own namespace so a failure in
one does not abort the rest.
"""
import os, runpy, sys, traceback, pathlib

HERE = pathlib.Path(__file__).parent / "audit_probes"
PROBES = [
    "th02_bracket_and_bundle.py",
    "th03_modelsel_robustness.py",
    "imp_edge_stationarity_probe.py",
    "imp_modelselection_probe.py",
]

run_dir = os.environ.get("LAB_RUN_DIR", ".")
out_path = pathlib.Path(run_dir) / "audit_probe_output.txt"
buf = []

class Tee:
    def write(self, s):
        buf.append(s); sys.__stdout__.write(s)
    def flush(self):
        sys.__stdout__.flush()

sys.stdout = Tee()
for p in PROBES:
    print("\n" + "#" * 78)
    print("#### PROBE: %s" % p)
    print("#" * 78)
    try:
        runpy.run_path(str(HERE / p), run_name="__main__")
    except Exception:
        print("!!! PROBE FAILED:\n" + traceback.format_exc())
sys.stdout = sys.__stdout__

out_path.write_text("".join(buf))
print("\nwrote %s (%d chars)" % (out_path, len("".join(buf))))
