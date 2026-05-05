"""RunRecord v1 — single source of truth for sim run metadata.

run_id = sha256[:16] of canonical JSON (sorted keys, relative paths, no created_at).
Same inputs → same run_id across machines and reruns.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


CONFOUNDS_CHECKLIST: List[str] = [
    "measurement_invariance_checked",
    "multiple_comparisons_corrected",
    "effect_size_reported",
    "confidence_intervals_reported",
    "null_model_compared",
    "data_leakage_checked",
    "reproducibility_seed_fixed",
    "outlier_handling_documented",
]


@dataclass
class RunRecord:
    run_id: str
    created_at: str                    # ISO-8601; excluded from ID hash
    mode: str                          # "psi" | "meditation"
    repo: str
    git_commit: str
    argv: List[str]

    input: Dict[str, Any]             # mode-specific input params
    metrics: Dict[str, float]         # I_mean, I_std, I_final, vort_mean, n_steps,
                                      # Qz_mean, Qabs_mean, f_dress
    artifacts: Dict[str, str]         # relative paths: md_path, json_path

    confounds: List[str]              # subset of CONFOUNDS_CHECKLIST that apply / are satisfied
    guardrails: Dict[str, Any]        # thresholds + pass/fail flags
    h8_falsifiers: List[Dict[str, str]]  # [{prediction, discriminator, status}]

    notes: str = ""
    schema_version: str = "1"

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "mode": self.mode,
            "repo": self.repo,
            "git_commit": self.git_commit,
            "argv": self.argv,
            "input": self.input,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "confounds": self.confounds,
            "guardrails": self.guardrails,
            "h8_falsifiers": self.h8_falsifiers,
            "notes": self.notes,
        }


def canonicalize_paths(d: Dict[str, Any], repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Return a copy of d with any Path values converted to relative POSIX strings."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, Path):
            if repo_root is not None:
                try:
                    v = v.relative_to(repo_root)
                except ValueError:
                    pass
            out[k] = v.as_posix()
        elif isinstance(v, dict):
            out[k] = canonicalize_paths(v, repo_root)
        elif isinstance(v, list):
            out[k] = [
                canonicalize_paths(i, repo_root) if isinstance(i, dict) else
                (i.as_posix() if isinstance(i, Path) else i)
                for i in v
            ]
        else:
            out[k] = v
    return out


def build_run_id(
    mode: str,
    repo: str,
    git_commit: str,
    argv: List[str],
    input_params: Dict[str, Any],
    metrics: Dict[str, float],
    repo_root: Optional[Path] = None,
) -> str:
    """Stable sha256[:16] hash — does NOT include created_at or artifact paths."""
    canonical = {
        "mode": mode,
        "repo": repo,
        "git_commit": git_commit,
        "argv": argv,
        "input": canonicalize_paths(input_params, repo_root),
        "metrics": {k: metrics[k] for k in sorted(metrics)},
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# JSON schema-like dict for downstream validation
RUN_RECORD_SCHEMA: Dict[str, Any] = {
    "$schema": "run_record_v1",
    "required": [
        "schema_version", "run_id", "created_at", "mode", "repo",
        "git_commit", "argv", "input", "metrics", "artifacts",
        "confounds", "guardrails", "h8_falsifiers",
    ],
    "properties": {
        "schema_version": {"type": "string", "const": "1"},
        "run_id": {"type": "string", "minLength": 16, "maxLength": 16},
        "created_at": {"type": "string"},
        "mode": {"type": "string", "enum": ["psi", "meditation"]},
        "repo": {"type": "string"},
        "git_commit": {"type": "string"},
        "argv": {"type": "array", "items": {"type": "string"}},
        "input": {"type": "object"},
        "metrics": {
            "type": "object",
            "required": ["I_mean", "I_std", "I_final", "vort_mean", "n_steps",
                         "Qz_mean", "Qabs_mean", "f_dress"],
        },
        "artifacts": {
            "type": "object",
            "required": ["md_path", "json_path"],
        },
        "confounds": {"type": "array", "items": {"type": "string"}},
        "guardrails": {"type": "object"},
        "h8_falsifiers": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["prediction", "discriminator", "status"],
            },
        },
    },
}


def validate_run_record_dict(d: Dict[str, Any]) -> List[str]:
    """Return a list of validation errors (empty = valid)."""
    errors: List[str] = []
    required = RUN_RECORD_SCHEMA["required"]
    for key in required:
        if key not in d:
            errors.append(f"Missing required field: {key}")
    if "mode" in d and d["mode"] not in ("psi", "meditation"):
        errors.append(f"Invalid mode: {d['mode']!r}")
    if "metrics" in d:
        for mkey in RUN_RECORD_SCHEMA["properties"]["metrics"]["required"]:
            if mkey not in d["metrics"]:
                errors.append(f"Missing metric: {mkey}")
    if "artifacts" in d:
        for akey in ("md_path", "json_path"):
            if akey not in d["artifacts"]:
                errors.append(f"Missing artifact path: {akey}")
    return errors
