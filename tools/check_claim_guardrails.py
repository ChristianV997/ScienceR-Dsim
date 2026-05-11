#!/usr/bin/env python3
"""Scan repository text files for unsafe BTC/ICFT claim language.

This stdlib-only utility is intentionally conservative. It is a lightweight
operator check for generated reports, docs, configs, and code comments. It does
not evaluate science, download data, run models, or promote claims.

Default behavior skips generated caches, VCS metadata, large binary-like files,
and this script itself because it must contain the guarded phrase registry.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


# Keep this registry centralized for operator checks. The script excludes itself
# by default so the registry can be stored here without causing self-failures.
BANNED_PHRASES: tuple[str, ...] = (
    "proves consciousness",
    "consciousness proven",
    "soul proven",
    "afterlife proven",
    "liberation detected",
    "ontology solved",
    "ultimate reality",
    "q equals self",
    "q equals soul",
    "q_abs equals suffering",
    "f_dress equals karma",
)

DEFAULT_EXTENSIONS: tuple[str, ...] = (
    ".md",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".csv",
)

DEFAULT_EXCLUDE_PARTS: tuple[str, ...] = (
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "outputs",
    "data",
    "datasets",
    "artifacts",
)

DEFAULT_EXCLUDE_PATHS: tuple[str, ...] = (
    "tools/check_claim_guardrails.py",
)


@dataclass(frozen=True)
class GuardrailHit:
    path: str
    line: int
    phrase: str
    excerpt: str


def _normalized_path(path: Path) -> str:
    try:
        return path.as_posix()
    except Exception:
        return str(path)


def _should_skip(path: Path, root: Path, extensions: set[str], include_outputs: bool) -> bool:
    rel = path.relative_to(root)
    rel_text = rel.as_posix()
    if rel_text in DEFAULT_EXCLUDE_PATHS:
        return True
    parts = set(rel.parts)
    excludes = set(DEFAULT_EXCLUDE_PARTS)
    if include_outputs:
        excludes.discard("outputs")
    if parts.intersection(excludes):
        return True
    if path.suffix.lower() not in extensions:
        return True
    return False


def _iter_files(root: Path, extensions: set[str], include_outputs: bool) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if _should_skip(path, root, extensions, include_outputs):
            continue
        yield path


def _scan_file(path: Path, root: Path, phrases: tuple[str, ...]) -> list[GuardrailHit]:
    hits: list[GuardrailHit] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return hits
    except OSError as exc:
        raise RuntimeError(f"Failed reading {path}: {exc}") from exc

    rel = _normalized_path(path.relative_to(root) if root.is_dir() else path)
    for line_no, line in enumerate(text.splitlines(), start=1):
        lower = line.lower()
        for phrase in phrases:
            if phrase in lower:
                hits.append(
                    GuardrailHit(
                        path=rel,
                        line=line_no,
                        phrase=phrase,
                        excerpt=line.strip()[:240],
                    )
                )
    return hits


def scan_paths(
    root: Path,
    *,
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
    phrases: tuple[str, ...] = BANNED_PHRASES,
    include_outputs: bool = False,
) -> list[GuardrailHit]:
    root = root.resolve()
    extension_set = {ext if ext.startswith(".") else f".{ext}" for ext in extensions}
    hits: list[GuardrailHit] = []
    for path in _iter_files(root, extension_set, include_outputs):
        hits.extend(_scan_file(path.resolve(), root, phrases))
    return hits


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan text files for unsafe BTC/ICFT claim language.")
    parser.add_argument("paths", nargs="*", default=["."], help="Files or directories to scan.")
    parser.add_argument(
        "--extension",
        action="append",
        dest="extensions",
        help="File extension to include. May be provided multiple times. Defaults to common text/code formats.",
    )
    parser.add_argument(
        "--include-outputs",
        action="store_true",
        help="Include outputs/ directories. By default generated outputs are skipped.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv or sys.argv[1:]))
    extensions = tuple(args.extensions or DEFAULT_EXTENSIONS)
    all_hits: list[GuardrailHit] = []
    errors: list[str] = []

    for raw_path in args.paths:
        root = Path(raw_path)
        if not root.exists():
            errors.append(f"Path does not exist: {raw_path}")
            continue
        try:
            all_hits.extend(scan_paths(root, extensions=extensions, include_outputs=args.include_outputs))
        except RuntimeError as exc:
            errors.append(str(exc))

    ok = not all_hits and not errors
    result = {
        "ok": ok,
        "hits": [asdict(hit) for hit in all_hits],
        "errors": errors,
        "n_hits": len(all_hits),
        "n_errors": len(errors),
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
        if all_hits:
            for hit in all_hits:
                print(f"{hit.path}:{hit.line}: guarded phrase '{hit.phrase}' :: {hit.excerpt}")
        if ok:
            print("PASS: no guarded claim language found")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
