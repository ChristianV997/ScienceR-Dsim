from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment

class WorldlineTracker:
    """Track defect trajectories over time using Hungarian assignment."""

    def __init__(self, max_dist: float = 3.0):
        self.max_dist = float(max_dist)
        self.lines = {}
        self.prev_pos = None
        self.prev_ids = []
        self.next_id = 0

    def update(self, defects: np.ndarray, t: float):
        """Associate current defects to previous tracks and update worldlines."""
        if defects is None or len(defects) == 0:
            self.prev_pos = None
            self.prev_ids = []
            return
        curr_pos = np.asarray(defects)[:, :3].astype(float)

        if self.prev_pos is None or len(self.prev_pos) == 0:
            self.prev_ids = []
            for p in curr_pos:
                lid = self.next_id
                self.next_id += 1
                self.lines[lid] = [(float(p[0]), float(p[1]), float(p[2]), float(t))]
                self.prev_ids.append(lid)
            self.prev_pos = curr_pos
            return

        cost = np.linalg.norm(self.prev_pos[:, None, :] - curr_pos[None, :, :], axis=2)
        row, col = linear_sum_assignment(cost)
        assigned_curr = set()
        prev_id_map = {idx: lid for idx, lid in enumerate(self.prev_ids)}
        curr_ids = [None] * len(curr_pos)

        for r, c in zip(row, col):
            if cost[r, c] <= self.max_dist:
                lid = prev_id_map[r]
                p = curr_pos[c]
                self.lines.setdefault(lid, []).append((float(p[0]), float(p[1]), float(p[2]), float(t)))
                assigned_curr.add(c)
                curr_ids[c] = lid

        for c, p in enumerate(curr_pos):
            if c not in assigned_curr:
                lid = self.next_id
                self.next_id += 1
                self.lines[lid] = [(float(p[0]), float(p[1]), float(p[2]), float(t))]
                curr_ids[c] = lid

        self.prev_pos = curr_pos
        self.prev_ids = curr_ids if len(curr_pos) else []

    def get(self):
        """Return worldlines as a mapping from track id to sampled 4D points."""
        return self.lines
