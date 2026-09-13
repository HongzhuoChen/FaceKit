"""Identity embeddings for the privacy audit.

Ported from ``face_audit.compute_embeddings`` / ``face_audit.utils``. Faces
are found with InsightFace (buffalo_l), with the same escalating-resize
fallback as the original; images without a detectable face get an all-zero
row, which the metrics drop. Embeddings are cached per (backbone, split,
cohort) as ``.npy`` + ``.json`` so a re-run skips finished folders.

Backbones
---------
arcface : InsightFace buffalo_l recognition head (512-D). Weights download
          automatically to ``~/.insightface`` on first use.
adaface : IR-50 / IR-101 from https://github.com/mk-minchul/AdaFace; needs the
          repository directory (for ``net.py``) and a checkpoint.
lvface  : LVFace ONNX model (https://huggingface.co/bytedance-research/LVFace).

One deliberate difference from the original: for AdaFace and LVFace the
5-point alignment is applied to the same (possibly resized) image the
keypoints were detected on. The original applied keypoints found on a
resized copy to the full-size image, which shifted the crop whenever the
detector fell back to a resized input.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

os.environ.setdefault("ORT_DISABLE_CPU_AFFINITY", "1")

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

# Standard ArcFace 112x112 alignment template (5 landmarks).
TEMPLATE_112 = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


def list_images(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def load_bgr(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        from PIL import Image
        pil = Image.open(path)
        if pil.mode == "RGBA":
            bg = Image.new("RGB", pil.size, (255, 255, 255))
            bg.paste(pil, mask=pil.split()[3])
            pil = bg
        return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def _resize_safe(img: np.ndarray, target: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = target / max(h, w)
    return cv2.resize(img, (max(32, int(w * s) // 32 * 32), max(32, int(h * s) // 32 * 32)))


def detect_face(face_analysis, img: np.ndarray):
    """First InsightFace detection, trying several input sizes like the original."""
    for candidate in (lambda: _resize_safe(img, 224), lambda: img,
                      lambda: _resize_safe(img, 320), lambda: _resize_safe(img, 512),
                      lambda: _resize_safe(img, 800)):
        try:
            faces = face_analysis.get(candidate())
        except Exception:
            continue
        if faces:
            return faces[0], candidate()
    return None, None


def align_112(img: np.ndarray, kps: np.ndarray) -> Optional[np.ndarray]:
    if kps is None or kps.shape != (5, 2):
        return None
    m, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), TEMPLATE_112, method=cv2.LMEDS)
    if m is None:
        return None
    return cv2.warpAffine(img, m, (112, 112), borderValue=0)


@dataclass
class Backbone:
    name: str
    dim: int
    embed: Callable[[np.ndarray], Optional[np.ndarray]]  # BGR image -> vector or None


def make_face_analysis(device: str, det_size: int = 224, threads: Optional[int] = None):
    """InsightFace buffalo_l detector + ArcFace head.

    ``threads`` caps onnxruntime's intra-op threads; InsightFace offers no
    way to pass session options, so its InferenceSession call is wrapped.
    Without the cap onnxruntime spins one thread per core, which on a busy
    shared node makes each image take tens of seconds.
    """
    import onnxruntime as ort
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import model_zoo

    ort.set_default_logger_severity(4)
    cls = model_zoo.PickableInferenceSession  # subclass of ort.InferenceSession
    original_init = cls.__init__
    if threads:
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1

        def patched_init(self, model_path, **kwargs):
            kwargs.setdefault("sess_options", so)
            original_init(self, model_path, **kwargs)

        cls.__init__ = patched_init
    try:
        app = FaceAnalysis(name="buffalo_l")
    finally:
        cls.__init__ = original_init
    app.prepare(ctx_id=0 if device.startswith("cuda") else -1, det_size=(det_size, det_size))
    return app


def arcface_backbone(face_analysis) -> Backbone:
    def embed(img):
        face, _ = detect_face(face_analysis, img)
        return None if face is None else np.asarray(face.embedding, dtype=np.float32)
    return Backbone("arcface", 512, embed)


def adaface_backbone(face_analysis, repo_dir: Path, ckpt: Path, device: str,
                     model_type: str = "ir_50") -> Backbone:
    import torch

    repo_dir = Path(repo_dir).resolve()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    import net as adaface_net  # from the AdaFace repository

    model = adaface_net.build_model(model_type)
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    model.load_state_dict({k[6:]: v for k, v in state.items() if k.startswith("model.")}, strict=True)
    model = model.to(device).eval()

    def embed(img):
        face, used = detect_face(face_analysis, img)
        if face is None:
            return None
        aligned = align_112(used, face.kps)
        if aligned is None:
            return None
        x = (aligned.astype(np.float32) / 255.0 - 0.5) / 0.5  # BGR, as in the original
        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.no_grad():
            feat, _ = model(x)
        return feat[0].cpu().numpy().astype(np.float32)
    return Backbone("adaface", 512, embed)


def lvface_backbone(face_analysis, onnx_path: Path, device: str) -> Backbone:
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.startswith("cuda") \
        else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(Path(onnx_path).resolve()), providers=providers)
    inp, out = sess.get_inputs()[0].name, sess.get_outputs()[0].name

    def run(batch):
        r = np.asarray(sess.run([out], {inp: batch})[0])
        return r.reshape(batch.shape[0], -1)

    dim = int(run(np.zeros((1, 3, 112, 112), np.float32)).shape[1])

    def embed(img):
        face, used = detect_face(face_analysis, img)
        if face is None:
            return None
        aligned = align_112(used, face.kps)
        if aligned is None:
            return None
        rgb = cv2.cvtColor(cv2.resize(aligned, (112, 112)), cv2.COLOR_BGR2RGB)
        x = ((rgb.transpose(2, 0, 1) / 255.0) - 0.5) / 0.5
        return run(x[None].astype(np.float32))[0].astype(np.float32)
    return Backbone("lvface", dim, embed)


def embed_folder(backbone: Backbone, folder: Path, cache_prefix: Path,
                 progress: Optional[Callable[[int], None]] = None) -> Tuple[np.ndarray, List[str]]:
    """Embed every image in ``folder``; zero rows mark undetected faces.

    Results are written to ``<cache_prefix>.npy`` and ``<cache_prefix>.json``
    and read back on the next call if the file list is unchanged.
    """
    files = list_images(folder)
    names = [p.name for p in files]
    npy, meta = cache_prefix.with_suffix(".npy"), cache_prefix.with_suffix(".json")
    if npy.exists() and meta.exists():
        info = json.loads(meta.read_text())
        if info.get("filenames") == names and info.get("backbone") == backbone.name:
            return np.load(npy), names

    emb = np.zeros((len(files), backbone.dim), dtype=np.float32)
    no_face = []
    for i, path in enumerate(files):
        img = load_bgr(path)
        vec = None if img is None else backbone.embed(img)
        if vec is None:
            no_face.append(path.name)
        else:
            emb[i] = vec
        if progress:
            progress(1)

    cache_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy, emb)
    meta.write_text(json.dumps({
        "backbone": backbone.name, "source_folder": str(folder), "filenames": names,
        "num_images": len(names), "no_face_count": len(no_face), "no_face_files": no_face,
        "embedding_dim": backbone.dim,
    }, indent=2))
    return emb, names
