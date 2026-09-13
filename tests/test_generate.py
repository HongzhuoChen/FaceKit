"""`facekit generate` on a tiny randomly initialised StyleGAN3 generator.

Covers the vendor sys.path wiring, network-pickle loading through the
vendored ``legacy`` module, the seed -> image contract, and the CLI's
output layout. No pretrained weights are needed.
"""
from __future__ import annotations

import pickle

import pytest

torch = pytest.importorskip("torch")

from typer.testing import CliRunner  # noqa: E402

from facekit.cli import app  # noqa: E402
from facekit.core.synth.generate import generate, load_generator, parse_seeds  # noqa: E402
from facekit.vendor import ensure_stylegan3_on_path  # noqa: E402


@pytest.fixture(scope="module")
def tiny_pkl(tmp_path_factory):
    ensure_stylegan3_on_path()
    from training.networks_stylegan3 import Generator  # vendored

    torch.manual_seed(0)
    G = Generator(
        z_dim=16, c_dim=0, w_dim=16, img_resolution=32, img_channels=3,
        mapping_kwargs=dict(num_layers=2),
        channel_base=2048, channel_max=32, num_layers=6, num_critical=2,
    )
    path = tmp_path_factory.mktemp("net") / "network-snapshot-000001.pkl"
    with open(path, "wb") as f:
        pickle.dump({"G": G, "D": torch.nn.Identity(), "G_ema": G}, f)
    return path


def test_parse_seeds():
    assert parse_seeds("0,3,10-12") == [0, 3, 10, 11, 12]
    with pytest.raises(ValueError):
        parse_seeds("")


def test_seed_determines_image(tiny_pkl, tmp_path):
    G = load_generator(tiny_pkl, torch.device("cpu"))
    a = list(generate(G, [5], tmp_path / "a"))
    b = list(generate(G, [5], tmp_path / "b"))
    assert a[0].name == "seed0005.png"
    assert a[0].read_bytes() == b[0].read_bytes()


def test_class_idx_rejected_for_unconditional(tiny_pkl, tmp_path):
    G = load_generator(tiny_pkl, torch.device("cpu"))
    with pytest.raises(ValueError, match="unconditional"):
        list(generate(G, [0], tmp_path, class_idx=0))


def test_cli_layout(tiny_pkl, tmp_path):
    out = tmp_path / "syn"
    r = CliRunner().invoke(app, [
        "generate", "--network", str(tiny_pkl), "-o", str(out),
        "--name", "cohort_x", "--n", "3", "--first-seed", "10", "--device", "cpu",
    ])
    assert r.exit_code == 0, r.output
    assert sorted(p.name for p in (out / "cohort_x").iterdir()) == [
        "seed0010.png", "seed0011.png", "seed0012.png",
    ]


def test_cli_requires_n_xor_seeds(tiny_pkl, tmp_path):
    r = CliRunner().invoke(app, ["generate", "--network", str(tiny_pkl), "-o", str(tmp_path)])
    assert r.exit_code == 2
    r = CliRunner().invoke(app, ["generate", "--network", str(tiny_pkl), "-o", str(tmp_path),
                                 "--n", "1", "--seeds", "0"])
    assert r.exit_code == 2
