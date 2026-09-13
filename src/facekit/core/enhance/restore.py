"""GFPGAN v1.4 face restoration, following the upstream ``GFPGANer`` helper:
facexlib detects and aligns each face to 512x512, GFPGAN restores it, and
the result is pasted back onto the (upscaled) input image."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from facekit.core.enhance.colorize import CACHE_DIR

GFPGAN_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"


def gfpgan_weights() -> Path:
    path = CACHE_DIR / "gfpgan" / "GFPGANv1.4.pth"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(GFPGAN_URL, str(path), progress=True)
    return path


class Restorer:
    """``restore(bgr) -> bgr`` at ``upscale`` times the input size."""

    def __init__(self, device: str = "cpu", upscale: int = 2, weight: float = 0.5):
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        from facekit.vendor.gfpgan import GFPGANv1Clean

        self.device = torch.device(device)
        self.weight = weight
        self.net = GFPGANv1Clean(out_size=512, num_style_feat=512, channel_multiplier=2,
                                 decoder_load_path=None, fix_decoder=False, num_mlp=8,
                                 input_is_latent=True, different_w=True, narrow=1, sft_half=True)
        state = torch.load(str(gfpgan_weights()), map_location="cpu", weights_only=True)
        self.net.load_state_dict(state["params_ema" if "params_ema" in state else "params"], strict=True)
        self.net.eval().to(self.device)
        self.helper = FaceRestoreHelper(
            upscale, face_size=512, crop_ratio=(1, 1), det_model="retinaface_resnet50",
            save_ext="png", use_parse=True, device=self.device,
            model_rootpath=str(CACHE_DIR / "facexlib"),
        )

    @torch.no_grad()
    def restore(self, bgr: np.ndarray) -> np.ndarray:
        """Returns the restored image, or the plain upscaled image when
        facexlib finds no face."""
        h = self.helper
        h.clean_all()
        h.read_image(bgr)
        h.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
        h.align_warp_face()
        for face in h.cropped_faces:
            x = torch.from_numpy(cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            x = ((x.permute(2, 0, 1) - 0.5) / 0.5).unsqueeze(0).to(self.device)
            try:
                out = self.net(x, return_rgb=False, weight=self.weight)[0]
                out = (out.squeeze(0).clamp(-1, 1) + 1) / 2 * 255.0
                restored = cv2.cvtColor(out.permute(1, 2, 0).round().cpu().numpy().astype(np.uint8),
                                        cv2.COLOR_RGB2BGR)
            except RuntimeError:
                restored = face
            h.add_restored_face(restored)
        h.get_inverse_affine(None)
        return h.paste_faces_to_input_image(upsample_img=None)
