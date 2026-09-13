"""Sample images from a StyleGAN3 generator pickle.

A port of the loop in the vendored ``gen_images.py``: one latent per seed,
drawn with ``numpy.random.RandomState(seed)`` so that a seed identifies an
image regardless of batch or device, mapped through ``G_ema`` with constant
noise, and written as ``seed{NNNN}.png``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import PIL.Image
import torch

from facekit.vendor import ensure_stylegan3_on_path

_RANGE_RE = re.compile(r"^(\d+)-(\d+)$")


def parse_seeds(spec: str) -> List[int]:
    """``'0,3,10-12'`` -> ``[0, 3, 10, 11, 12]`` (StyleGAN3 syntax, inclusive)."""
    seeds: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        m = _RANGE_RE.match(part)
        if m:
            seeds.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        elif part:
            seeds.append(int(part))
    if not seeds:
        raise ValueError(f"no seeds in {spec!r}")
    return seeds


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def load_generator(network_pkl: Path, device: torch.device):
    """Return ``G_ema`` from a StyleGAN3 network pickle, in eval mode."""
    ensure_stylegan3_on_path()
    import legacy  # noqa: E402  (vendored)

    with open(network_pkl, "rb") as f:
        G = legacy.load_network_pkl(f)["G_ema"]
    return G.eval().requires_grad_(False).to(device)


def generate(
    G,
    seeds: List[int],
    out_dir: Path,
    truncation_psi: float = 1.0,
    class_idx: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Iterator[Path]:
    """Write one PNG per seed into ``out_dir`` and yield each path."""
    if device is None:
        device = next(G.parameters()).device
    out_dir.mkdir(parents=True, exist_ok=True)

    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise ValueError(
                f"conditional generator with {G.c_dim} classes: --class-idx is required"
            )
        if not 0 <= class_idx < G.c_dim:
            raise ValueError(f"--class-idx {class_idx} out of range [0, {G.c_dim})")
        label[:, class_idx] = 1
    elif class_idx is not None:
        raise ValueError("--class-idx given but the generator is unconditional")

    for seed in seeds:
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode="const")
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        path = out_dir / f"seed{seed:04d}.png"
        PIL.Image.fromarray(img[0].cpu().numpy(), "RGB").save(path)
        yield path
