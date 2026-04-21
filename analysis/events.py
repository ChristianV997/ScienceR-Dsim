from __future__ import annotations
import pandas as pd

def detect_events(df: pd.DataFrame):
    events = []
    for z, sub in df.groupby("z"):
        sub = sub.sort_values("t")
        q = sub["Q"].values
        t = sub["t"].values
        for i in range(len(q) - 1):
            if q[i + 1] != q[i]:
                events.append({"z": int(z), "t": float(t[i]), "delta": int(q[i + 1] - q[i])})
    return pd.DataFrame(events)
