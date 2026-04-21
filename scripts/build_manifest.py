from __future__ import annotations
from pathlib import Path
import json

ROOT = Path(".")
EXCLUDE = {"results", "__pycache__", ".git", ".venv"}

def main():
    files = []
    for p in ROOT.rglob("*"):
        if any(part in EXCLUDE for part in p.parts):
            continue
        if p.is_file():
            files.append(str(p))
    Path("project_manifest.json").write_text(json.dumps({"files": files}, indent=2), encoding="utf-8")
    print("Wrote project_manifest.json")

if __name__ == "__main__":
    main()
