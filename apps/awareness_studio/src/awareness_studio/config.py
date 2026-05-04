import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

APP_ROOT: Path = Path(__file__).resolve().parent.parent.parent  # apps/awareness_studio/
INPUTS_DIR: Path = APP_ROOT / "inputs" / "notion_export"
INDEX_DIR: Path = APP_ROOT / ".index"

# Chunking
MAX_CHUNK_CHARS: int = 1200
CHUNK_OVERLAP_CHARS: int = 150

# BM25
BM25_K1: float = 1.5
BM25_B: float = 0.75
DEFAULT_TOP_K: int = 8

# LLM
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic")
LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
LLM_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
