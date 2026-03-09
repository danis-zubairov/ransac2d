"""Examples for ransac2d.

Run from the repository root, for example:

- ``python -m examples.demo_rectangle``
- ``PYTHONPATH=src python examples/demo_rectangle.py``
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_src_on_path() -> None:
    """Ensure ``src`` directory is on ``sys.path`` when running examples directly."""
    src = Path(__file__).resolve().parents[1] / "src"
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
