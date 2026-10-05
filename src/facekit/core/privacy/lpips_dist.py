"""Perceptual (LPIPS) nearest-neighbour distances for the privacy audit.

Ported from ``face_audit.compute_lpips``. Every query image is compared
exhaustively against every reference image and the smallest distance kept.
The original looped over pairs one at a time; here the reference side is
batched, which changes nothing numerically. Arrays are cached as ``.npy``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from PIL import Image


def make_lpips(net: str, device: str):
    import lpips

    return lpips.LPIPS(net=net, verbose=False).to(device).eval()


def load_tensors(paths: List[Path], size: int, device: str) -> torch.Tensor:
    """(N, 3, size, size) in [-1, 1]; unreadable images become NaN planes."""
    out = torch.full((len(paths), 3, size, size), float("nan"), device=device)
    for i, p in enumerate(paths):
        try:
            img = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
        except Exception:
            continue
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        out[i] = (x * 2.0 - 1.0).to(device)
    return out


@torch.no_grad()
def min_dists_cross(query: torch.Tensor, ref: torch.Tensor, loss_fn, batch: int = 64,
                    progress: Optional[Callable[[int], None]] = None) -> np.ndarray:
    """Smallest LPIPS from each query image to any reference image."""
    valid_ref = ref[~torch.isnan(ref).flatten(1).any(1)]
    out = np.full(len(query), np.nan)
    for i in range(len(query)):
        q = query[i]
        if torch.isnan(q).any() or len(valid_ref) == 0:
            if progress:
                progress(1)
            continue
        best = float("inf")
        for j in range(0, len(valid_ref), batch):
            r = valid_ref[j:j + batch]
            d = loss_fn(q.unsqueeze(0).expand(len(r), -1, -1, -1), r).flatten()
            best = min(best, float(d.min()))
        out[i] = best
        if progress:
            progress(1)
    return out


@torch.no_grad()
def min_dists_within(x: torch.Tensor, loss_fn, batch: int = 64,
                     progress: Optional[Callable[[int], None]] = None) -> np.ndarray:
    """Smallest LPIPS from each image to any *other* image of the same set."""
    out = np.full(len(x), np.nan)
    nan_mask = torch.isnan(x).flatten(1).any(1)
    for i in range(len(x)):
        if nan_mask[i]:
            if progress:
                progress(1)
            continue
        keep = ~nan_mask.clone()
        keep[i] = False
        others = x[keep]
        if len(others) == 0:
            if progress:
                progress(1)
            continue
        best = float("inf")
        for j in range(0, len(others), batch):
            r = others[j:j + batch]
            d = loss_fn(x[i].unsqueeze(0).expand(len(r), -1, -1, -1), r).flatten()
            best = min(best, float(d.min()))
        out[i] = best
        if progress:
            progress(1)
    return out


def cached(path: Path, compute: Callable[[], np.ndarray]) -> np.ndarray:
    if path.exists():
        return np.load(path)
    arr = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr
