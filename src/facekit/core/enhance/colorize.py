"""DDColor colorization of grayscale photographs."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

CACHE_DIR = Path(os.environ.get("FACEKIT_CACHE_DIR", Path.home() / ".cache" / "facekit"))

# Hugging Face repositories published by the DDColor authors.
DDCOLOR_REPOS = {
    "large": "piddnad/ddcolor_modelscope",  # DDColor-L, the authors' default
    "tiny": "piddnad/ddcolor_paper_tiny",
}


def ddcolor_weights(model_size: str = "large") -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(DDCOLOR_REPOS[model_size], "pytorch_model.bin",
                                cache_dir=str(CACHE_DIR / "ddcolor")))


class Colorizer:
    """``colorize(bgr) -> bgr`` with DDColor at a fixed model input size."""

    def __init__(self, device: str = "cpu", model_size: str = "large", input_size: int = 512):
        from facekit.vendor.ddcolor import DDColor, ColorizationPipeline, build_ddcolor_model

        model = build_ddcolor_model(DDColor, model_path=str(ddcolor_weights(model_size)),
                                   input_size=input_size, model_size=model_size, device=device)
        self.pipeline = ColorizationPipeline(model, input_size=input_size, device=device)

    def colorize(self, bgr: np.ndarray) -> np.ndarray:
        if bgr.ndim == 2:
            bgr = np.repeat(bgr[:, :, None], 3, axis=2)
        return self.pipeline.process(bgr)
