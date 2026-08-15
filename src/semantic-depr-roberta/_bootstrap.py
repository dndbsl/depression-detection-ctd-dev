"""Import bootstrap for running scripts from the hyphenated project folder."""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_paths() -> None:
    """Add project and repository roots to sys.path."""
    project_root = Path(__file__).resolve().parent
    repo_root = project_root.parent
    for path in (project_root, repo_root):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)

