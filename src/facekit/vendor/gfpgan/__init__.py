"""GFPGAN face restoration (https://github.com/TencentARC/GFPGAN), Apache-2.0.

Vendored subset (commit 7552a77, copied 2026-09-13): ``gfpganv1_clean_arch.py``
and ``stylegan2_clean_arch.py``, with the ``basicsr`` registry decorators
removed and ``default_init_weights`` copied into ``arch_util.py``. The pip
package depends on ``basicsr``, whose setup.py does not run on Python 3.13 and
whose ``basicsr`` module name collides with DDColor's fork. Nothing else is
modified. Face detection and alignment for restoration come from ``facexlib``
(a pip dependency); the GFPGANv1.4 weights download from the GFPGAN GitHub
release on first use.
"""
from .gfpganv1_clean_arch import GFPGANv1Clean

__all__ = ["GFPGANv1Clean"]
