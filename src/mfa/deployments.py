"""Small JSON store for what each deploy created (resource names, agent IDs, versions).

Records live in ``.deployments/<agent>.<target>.json`` so redeploys update the
existing resource instead of creating a new one, and eval runs know what to call.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_DIR = Path(os.environ.get("MFA_DEPLOYMENTS_DIR", ".deployments"))


def _path(agent: str, target: str, root: Path | None = None) -> Path:
    return (root or DEFAULT_DIR) / f"{agent}.{target}.json"


def load(agent: str, target: str, root: Path | None = None) -> dict[str, Any] | None:
    p = _path(agent, target, root)
    return json.loads(p.read_text()) if p.exists() else None


def save(agent: str, target: str, record: dict[str, Any], root: Path | None = None) -> Path:
    p = _path(agent, target, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return p


def require(agent: str, target: str, root: Path | None = None) -> dict[str, Any]:
    rec = load(agent, target, root)
    if rec is None:
        raise RuntimeError(
            f"No deployment record for agent {agent!r} on {target!r}. "
            f"Run: mfa deploy {agent} --target {target}"
        )
    return rec
