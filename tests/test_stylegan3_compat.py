"""Does the vendored StyleGAN3 still run on the installed PyTorch?

StyleGAN3 targets PyTorch 1.9; the Python API has moved since. This test
replays the pieces of ``training_loop.py`` that do not need a GPU on a tiny
32x32 configuration built the way ``train.py --cfg stylegan3-t`` builds it:
dataset + InfiniteSampler + DataLoader, G / D / G_ema construction, the
augmentation pipe, every loss phase with gradfix enabled, the Adam optimizers
with train.py's kwargs, the EMA update, the snapshot pickle, and a reload
through ``legacy``. What it cannot exercise: CUDA plugins, fp16 layers, and
``conv2d_gradfix`` (which only activates on CUDA tensors).
"""
from __future__ import annotations

import copy
import pickle
import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("psutil")

from facekit.core.synth.pack import build_dataset_zip  # noqa: E402
from facekit.vendor import STYLEGAN3_DIR, ensure_stylegan3_on_path  # noqa: E402

ensure_stylegan3_on_path()
import dnnlib  # noqa: E402  (vendored)
import legacy  # noqa: E402
from torch_utils import misc  # noqa: E402
from torch_utils.ops import conv2d_gradfix, grid_sample_gradfix  # noqa: E402
from training import training_loop  # noqa: E402

RES, BATCH = 32, 4


@pytest.fixture(scope="module")
def dataset_zip(tmp_path_factory):
    import cv2

    root = tmp_path_factory.mktemp("ds")
    crops = root / "crops"
    crops.mkdir()
    rng = np.random.default_rng(0)
    for i in range(8):
        cv2.imwrite(str(crops / f"{i}.png"), rng.integers(0, 255, (RES, RES, 3), dtype=np.uint8))
    zip_path = root / "tiny.zip"
    build_dataset_zip(crops, zip_path)
    return zip_path


def _config(dataset_zip):
    """The relevant subset of what train.py assembles for --cfg stylegan3-t."""
    c = dnnlib.EasyDict()
    c.training_set_kwargs = dnnlib.EasyDict(
        class_name="training.dataset.ImageFolderDataset", path=str(dataset_zip),
        use_labels=False, max_size=None, xflip=True, resolution=RES, random_seed=0)
    c.G_kwargs = dnnlib.EasyDict(
        class_name="training.networks_stylegan3.Generator", z_dim=16, w_dim=16,
        mapping_kwargs=dnnlib.EasyDict(num_layers=8), channel_base=2048, channel_max=32,
        magnitude_ema_beta=0.5 ** (BATCH / (20 * 1e3)), num_layers=6, num_critical=2,
        num_fp16_res=0, conv_clamp=None)  # fp32, as train.py --fp32
    c.D_kwargs = dnnlib.EasyDict(
        class_name="training.networks_stylegan2.Discriminator", channel_base=2048, channel_max=32,
        block_kwargs=dnnlib.EasyDict(freeze_layers=0), mapping_kwargs=dnnlib.EasyDict(),
        epilogue_kwargs=dnnlib.EasyDict(mbstd_group_size=4), num_fp16_res=0, conv_clamp=None)
    c.G_opt_kwargs = dnnlib.EasyDict(class_name="torch.optim.Adam", betas=[0.0, 0.99], eps=1e-8, lr=0.0025)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name="torch.optim.Adam", betas=[0.0, 0.99], eps=1e-8, lr=0.002)
    c.loss_kwargs = dnnlib.EasyDict(class_name="training.loss.StyleGAN2Loss", r1_gamma=2.0)
    c.augment_kwargs = dnnlib.EasyDict(
        class_name="training.augment.AugmentPipe", xflip=1, rotate90=1, xint=1, scale=1, rotate=1,
        aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
    return c


def test_train_py_uses_float_betas():
    src = (STYLEGAN3_DIR / "train.py").read_text()
    assert not re.search(r"betas=\[0,", src), "Adam betas must be floats on current PyTorch"


def test_plugin_init_falls_back_when_build_fails(monkeypatch):
    from torch_utils import custom_ops
    from torch_utils.ops import bias_act

    monkeypatch.setattr(bias_act, "_plugin", None)
    monkeypatch.setattr(bias_act, "_plugin_failed", False)

    def boom(**kwargs):
        raise RuntimeError("no nvcc")

    monkeypatch.setattr(custom_ops, "get_plugin", boom)
    with pytest.warns(UserWarning, match="reference implementation"):
        assert bias_act._init() is False
    assert bias_act._init() is False  # not retried
    x = torch.randn(2, 3)
    assert torch.allclose(bias_act.bias_act(x, act="linear"), x)


def test_one_training_step_on_cpu(dataset_zip, tmp_path):
    c = _config(dataset_zip)
    device = torch.device("cpu")
    torch.manual_seed(0)
    np.random.seed(0)
    conv2d_gradfix.enabled = True
    grid_sample_gradfix.enabled = True

    # Data, as in training_loop.
    training_set = dnnlib.util.construct_class_by_name(**c.training_set_kwargs)
    sampler = misc.InfiniteSampler(dataset=training_set, rank=0, num_replicas=1, seed=0)
    loader = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=sampler, batch_size=BATCH))
    real_img, real_c = next(loader)
    assert real_img.shape == (BATCH, 3, RES, RES)

    # Networks.
    common = dict(c_dim=training_set.label_dim, img_resolution=RES, img_channels=3)
    G = dnnlib.util.construct_class_by_name(**c.G_kwargs, **common).train().requires_grad_(False).to(device)
    D = dnnlib.util.construct_class_by_name(**c.D_kwargs, **common).train().requires_grad_(False).to(device)
    G_ema = copy.deepcopy(G).eval()
    misc.print_module_summary(G, [torch.empty([BATCH, G.z_dim]), torch.empty([BATCH, G.c_dim])])
    misc.print_module_summary(D, [torch.empty([BATCH, 3, RES, RES]), torch.empty([BATCH, G.c_dim])])

    augment_pipe = dnnlib.util.construct_class_by_name(**c.augment_kwargs).train().requires_grad_(False).to(device)
    augment_pipe.p.copy_(torch.as_tensor(0.5))

    # Phases with lazy regularization for D only (G_reg_interval is None for stylegan3).
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, augment_pipe=augment_pipe, **c.loss_kwargs)
    phases = []
    for name, module, opt_kwargs, reg_interval in [("G", G, c.G_opt_kwargs, None), ("D", D, c.D_opt_kwargs, 16)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs)
            phases += [dnnlib.EasyDict(name=name + "both", module=module, opt=opt, interval=1)]
        else:
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs)
            phases += [dnnlib.EasyDict(name=name + "main", module=module, opt=opt, interval=1),
                       dnnlib.EasyDict(name=name + "reg", module=module, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.base_lr = phase.opt.param_groups[0]["lr"]

    # One iteration: every phase runs (batch_idx 0), lr schedule applied.
    factor = training_loop.lr_factor("cosine", 0, 1)
    real_img = (real_img.to(device).to(torch.float32) / 127.5 - 1).split(BATCH)
    real_c = real_c.to(device).split(BATCH)
    all_gen_z = torch.randn([len(phases) * BATCH, G.z_dim], device=device).split(BATCH)
    all_gen_c = torch.zeros([len(phases) * BATCH, G.c_dim], device=device).split(BATCH)
    for phase, gen_z, gen_c in zip(phases, all_gen_z, all_gen_c):
        for group in phase.opt.param_groups:
            group["lr"] = phase.base_lr * factor
        phase.opt.zero_grad(set_to_none=True)
        phase.module.requires_grad_(True)
        for ri, rc in zip(real_img, real_c):
            loss.accumulate_gradients(phase=phase.name, real_img=ri, real_c=rc,
                                      gen_z=gen_z.split(BATCH)[0], gen_c=gen_c.split(BATCH)[0],
                                      gain=phase.interval, cur_nimg=0)
        phase.module.requires_grad_(False)
        params = [p for p in phase.module.parameters() if p.grad is not None]
        assert params, phase.name
        for p in params:
            torch.nan_to_num(p.grad, nan=0, posinf=1e5, neginf=-1e5, out=p.grad)
        phase.opt.step()

    # EMA update and the augmentation-probability adjustment.
    ema_beta = 0.5 ** (BATCH / max(10 * 1000, 1e-8))
    for p_ema, p in zip(G_ema.parameters(), G.parameters()):
        p_ema.copy_(p.lerp(p_ema, ema_beta))
    for b_ema, b in zip(G_ema.buffers(), G.buffers()):
        b_ema.copy_(b)

    # Snapshot exactly as training_loop pickles it, then reload through legacy.
    snapshot = {"training_set_kwargs": dict(c.training_set_kwargs)}
    for name, module in [("G", G), ("D", D), ("G_ema", G_ema), ("augment_pipe", augment_pipe)]:
        module = copy.deepcopy(module).eval().requires_grad_(False)
        snapshot[name] = module
    pkl = tmp_path / "network-snapshot-000000.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(snapshot, f)
    with open(pkl, "rb") as f:
        data = legacy.load_network_pkl(f)
    img = data["G_ema"](torch.randn([1, G.z_dim]), torch.zeros([1, G.c_dim]), noise_mode="const")
    assert img.shape == (1, 3, RES, RES) and torch.isfinite(img).all()
