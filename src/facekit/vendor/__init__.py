"""Third-party code shipped inside FaceKit.

``stylegan3/`` is NVIDIA's StyleGAN3 (https://github.com/NVlabs/stylegan3),
under the NVIDIA Source Code License in ``stylegan3/LICENSE.txt``. It is not
covered by FaceKit's own license.

StyleGAN3 network pickles reference ``dnnlib`` and ``torch_utils`` as
top-level module names, and the StyleGAN3 scripts import each other the same
way, so the code cannot be relocated under ``facekit.vendor.stylegan3.*``.
Call :func:`ensure_stylegan3_on_path` before importing any of it.
"""
from __future__ import annotations

import sys
from pathlib import Path

STYLEGAN3_DIR = Path(__file__).resolve().parent / "stylegan3"


def ensure_stylegan3_on_path() -> Path:
    """Make ``dnnlib``, ``torch_utils``, ``training``, ``legacy`` importable."""
    p = str(STYLEGAN3_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    return STYLEGAN3_DIR
