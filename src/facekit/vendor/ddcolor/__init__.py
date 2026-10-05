"""DDColor image colorization (https://github.com/piddnad/DDColor), Apache-2.0.

Vendored subset (commit 2adb63f, copied 2026-09-13): ``ddcolor/model.py``,
``ddcolor/pipeline.py`` and ``basicsr/archs/ddcolor_arch_utils/*``, with the
``basicsr.archs.ddcolor_arch_utils`` imports rewritten to ``.arch_utils``.
The upstream package cannot be pip-installed next to GFPGAN: it ships its own
``basicsr`` fork under the same name, and its setup.py does not run on
Python 3.13. Nothing else is modified. Weights are downloaded from
Hugging Face (``piddnad/ddcolor_modelscope``) on first use.
"""
from .model import DDColor
from .pipeline import ColorizationPipeline, build_ddcolor_model

__all__ = ["DDColor", "ColorizationPipeline", "build_ddcolor_model"]
