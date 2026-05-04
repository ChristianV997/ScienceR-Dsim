import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from awareness_studio import config
from awareness_studio.chunking import chunk_documents
from awareness_studio.doc_schema import Chunk
from awareness_studio.io_markdown import load_documents
from awareness_studio.retrieval import BM25Index

_INDEX_FILE = config.INDEX_DIR / "chunks.json"
_bm25_cache: Optional[BM25Index] = None


def build_index(inputs_dir: Optional[Path] = None) -> BM25Index:
    global _bm25_cache
    inputs_dir = inputs_dir or config.INPUTS_DIR
    docs = load_documents(inputs_dir)
    chunks = chunk_documents(docs)
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(_INDEX_FILE, "w", encoding="utf-8") as fh:
        json.dump([asdict(c) for c in chunks], fh, ensure_ascii=False, indent=2)
    _bm25_cache = BM25Index(chunks)
    return _bm25_cache


def load_index(inputs_dir: Optional[Path] = None) -> BM25Index:
    global _bm25_cache
    if _bm25_cache is not None:
        return _bm25_cache
    if _INDEX_FILE.exists():
        with open(_INDEX_FILE, encoding="utf-8") as fh:
            raw = json.load(fh)
        chunks: List[Chunk] = [Chunk(**item) for item in raw]
        _bm25_cache = BM25Index(chunks)
        return _bm25_cache
    return build_index(inputs_dir)


def get_or_build_index(inputs_dir: Optional[Path] = None) -> BM25Index:
    return load_index(inputs_dir)


if __name__ == "__main__":
    idx = build_index()
    print(f"Built index with {len(idx.chunks)} chunks.", file=sys.stderr)
