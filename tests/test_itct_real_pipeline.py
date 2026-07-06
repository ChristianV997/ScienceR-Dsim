"""Tests for the real ITCT cessation pipeline (PR-3 scope; depends on PR-1's bids_ingest)."""
from __future__ import annotations

import importlib.util

import pytest

_HAVE = all(importlib.util.find_spec(m) for m in ("mne", "mne_bids", "edfio", "numpy", "scipy", "networkx", "ripser"))
pytestmark = pytest.mark.skipif(not _HAVE, reason="requires mne, mne-bids, edfio, scipy, networkx, ripser")


@pytest.fixture(scope="module")
def bids_root(tmp_path_factory):
    from tests.fixtures.make_synthetic_bids import build

    root = tmp_path_factory.mktemp("bids_synth")
    return str(build(str(root)))


def test_itct_real_mode_stamps_provenance(bids_root):
    from analysis.itct.itct_cessation_protocol_v3_full_stack import load_plv_from_bids, run

    plv_series, provenance = load_plv_from_bids(bids_root, "01", "sedated", n_windows=4)
    assert provenance == "real_bids"
    records = run(plv_series, provenance)
    assert records and all(r["provenance"] == "real_bids" for r in records)
    assert all("beta1" in r and "loschmidt_echo" in r for r in records)


def test_itct_synthetic_mode_stamps_provenance():
    from analysis.itct.itct_cessation_protocol_v3_full_stack import synthetic_plv_series, run

    plv_series, provenance = synthetic_plv_series(n_windows=6)
    assert provenance == "synthetic_proxy"
    records = run(plv_series, provenance)
    assert all(r["provenance"] == "synthetic_proxy" for r in records)
