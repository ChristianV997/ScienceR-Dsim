from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import main


class _Frame:
    def head(self):
        return self


def test_eeg_cli_forwards_topology_comparator_flags(monkeypatch):
    captured = {}

    def fake_run_eeg(*args, **kwargs):
        captured.update(kwargs)
        return _Frame()

    monkeypatch.setattr(main, "run_eeg", fake_run_eeg)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--mode",
            "eeg",
            "--compute-phase-grid-topology",
            "--compute-kuramoto",
            "--compute-leida",
        ],
    )

    main.main()

    assert captured["compute_phase_grid_topology"] is True
    assert captured["compute_kuramoto"] is True
    assert captured["compute_leida"] is True


def test_neural_mass_cli_forwards_simulation_options(monkeypatch):
    captured = {}

    def fake_run_neurolib(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(run_id="neural-mass-test")

    monkeypatch.setattr(main, "run_neurolib", fake_run_neurolib)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--mode",
            "neural_mass",
            "--n-nodes",
            "24",
            "--neurolib-model",
            "hopf",
            "--t-max",
            "5.5",
            "--coupling",
            "0.25",
            "--seed",
            "7",
        ],
    )

    main.main()

    assert captured == {
        "output_csv": "results/out.csv",
        "n_nodes": 24,
        "model_type": "hopf",
        "t_max": 5.5,
        "coupling": 0.25,
        "seed": 7,
    }


def test_fast_tr_cli_forwards_validation_options(monkeypatch):
    captured = {}

    def fake_run_fast_tr_validation(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(run_id="fast-tr-test")

    monkeypatch.setattr(main, "run_fast_tr_validation", fake_run_fast_tr_validation)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--mode",
            "fast_tr_validation",
            "--n-voxels",
            "12",
            "--n-timepoints",
            "120",
            "--tr",
            "0.72",
            "--seed",
            "11",
        ],
    )

    main.main()

    assert captured == {
        "output_csv": "results/out.csv",
        "n_voxels": 12,
        "n_timepoints": 120,
        "tr": 0.72,
        "seed": 11,
    }


@pytest.mark.parametrize(
    ("mode", "option"),
    [
        ("neural_mass", "--n-nodes"),
        ("neural_mass", "--t-max"),
        ("fast_tr_validation", "--n-voxels"),
        ("fast_tr_validation", "--n-timepoints"),
        ("fast_tr_validation", "--tr"),
    ],
)
def test_cli_rejects_nonpositive_dimensions(monkeypatch, mode, option):
    monkeypatch.setattr(sys, "argv", ["main.py", "--mode", mode, option, "0"])

    with pytest.raises(SystemExit, match="2"):
        main.main()
