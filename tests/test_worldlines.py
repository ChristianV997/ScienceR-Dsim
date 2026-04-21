from __future__ import annotations
import numpy as np
import pytest
from tracking.worldlines import WorldlineTracker


def test_empty_ndarray_update():
    t = WorldlineTracker()
    t.update(np.empty((0, 4)), 0.0)
    assert t.get() == {}


def test_none_update():
    t = WorldlineTracker()
    t.update(None, 0.0)
    assert t.get() == {}


def test_single_defect_creates_track():
    t = WorldlineTracker()
    d = np.array([[1.0, 2.0, 0.0, 1.0]])
    t.update(d, 0.0)
    lines = t.get()
    assert len(lines) == 1
    assert len(list(lines.values())[0]) == 1


def test_continuous_track():
    """A slowly-moving defect should produce a single worldline."""
    t = WorldlineTracker(max_dist=5.0)
    for step in range(5):
        d = np.array([[float(step) * 0.5, 0.0, 0.0, 1.0]])
        t.update(d, float(step))
    lines = t.get()
    assert len(lines) == 1
    assert len(list(lines.values())[0]) == 5


def test_far_jump_creates_new_track():
    """A jump beyond max_dist should create a new track."""
    t = WorldlineTracker(max_dist=1.0)
    t.update(np.array([[0.0, 0.0, 0.0, 1.0]]), 0.0)
    t.update(np.array([[50.0, 50.0, 0.0, 1.0]]), 1.0)
    lines = t.get()
    assert len(lines) == 2


def test_two_defects_tracked_separately():
    t = WorldlineTracker(max_dist=2.0)
    for step in range(3):
        d = np.array([
            [float(step), 0.0, 0.0, 1.0],
            [float(step), 10.0, 0.0, -1.0],
        ])
        t.update(d, float(step))
    lines = t.get()
    assert len(lines) == 2
    for pts in lines.values():
        assert len(pts) == 3


def test_get_returns_dict():
    t = WorldlineTracker()
    assert isinstance(t.get(), dict)
