#!/usr/bin/env bash
# Dual-engine topological benchmark driver.
#
# Runs both ingestion tracks (neuro + quantum) through the SAME topology/action
# instrumentation and emits a provenance-stamped JSON + side-by-side phase-space plots.
#
# Reality-first: with no real data present this uses clearly-labelled SYNTHETIC proxies
# (provenance "synthetic_proxy" / "quantum_field"). It asserts NO cross-domain equivalence
# and claims NO spectral-dimension convergence target. To run on REAL data, pass a BIDS
# root and/or an HDF5 quantum field via the environment variables documented below.
#
# Env (all optional):
#   BIDS_ROOT   -- path to a real BIDS EEG dataset (e.g. ds003969) for NeuroBIDSAdapter
#   QFIELD_H5   -- path to a real HDF5 quantum field grid for QuantumFieldAdapter
#   OUT_DIR     -- output directory (default: outputs/dual_engine)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
OUT_DIR="${OUT_DIR:-outputs/dual_engine}"

echo "== Dual-engine benchmark =="
echo "repo: ${REPO_ROOT}"
echo "out:  ${OUT_DIR}"

# 1) Synthetic collision benchmark (always runs; fully offline).
echo "-- Test 1+2: synthetic collision benchmark --"
python -m dual_engine.benchmark --out "${OUT_DIR}" --n-windows 10

# 2) Optional real neuro track.
if [[ -n "${BIDS_ROOT:-}" ]]; then
  echo "-- Real neuro track: ${BIDS_ROOT} --"
  python - <<PY
import json, sys
from pathlib import Path
from dual_engine.data_adapters import NeuroBIDSAdapter
from dual_engine.topology_engine import analyze_windows
from dual_engine.action_auditor import audit_field
a = NeuroBIDSAdapter("${BIDS_ROOT}")
f = a.load_su2_field(n_windows=8)
topo = [r.to_dict() for r in analyze_windows(f)]
aud = audit_field(f).to_dict()
out = Path("${OUT_DIR}") / "real_neuro_result.json"
out.write_text(json.dumps({"provenance": f.provenance, "topology": topo, "audit": aud}, indent=2))
print("real neuro provenance:", f.provenance, "-> ", out)
PY
else
  echo "   (skipped: set BIDS_ROOT to run the real neuro track)"
fi

# 3) Optional real quantum track.
if [[ -n "${QFIELD_H5:-}" ]]; then
  echo "-- Real quantum track: ${QFIELD_H5} --"
  python - <<PY
import json
from pathlib import Path
from dual_engine.data_adapters import QuantumFieldAdapter
from dual_engine.topology_engine import analyze_windows
from dual_engine.action_auditor import audit_field
a = QuantumFieldAdapter("${QFIELD_H5}")
f = a.load_su2_field()
topo = [r.to_dict() for r in analyze_windows(f)]
aud = audit_field(f).to_dict()
out = Path("${OUT_DIR}") / "real_quantum_result.json"
out.write_text(json.dumps({"provenance": f.provenance, "topology": topo, "audit": aud}, indent=2))
print("real quantum provenance:", f.provenance, "-> ", out)
PY
else
  echo "   (skipped: set QFIELD_H5 to run the real quantum track)"
fi

echo "== Done. Artifacts in ${OUT_DIR}/ =="
ls -la "${OUT_DIR}/" 2>/dev/null || true
