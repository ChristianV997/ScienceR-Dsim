"""Root conftest – ensures the repository root is on sys.path for all tests."""
from __future__ import annotations
import sys
from pathlib import Path

# Add the repo root so that bare ``from core.topology import ...`` imports work
# when pytest is invoked from any working directory.
sys.path.insert(0, str(Path(__file__).parent))
