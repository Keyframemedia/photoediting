"""Sky segmentation and replacement (in linear HDR, before tone mapping).

Segmentation: UperNet-ConvNeXt trained on ADE20K (MIT licence), run at ~640px.
The coarse mask is refined at full resolution with a luminance-keyed matte so
tree branches, fences and window mullions keep clean edges.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from .denoise import guided_filter
from .merge import luminance
from .tonemap import srgb_decode, srgb_encode

_MODEL = {}
SKY_CLASS = 2  # ADE20K
WINDOW_CLASSES = (8, 14)  # ADE20K windowpane, door (glass sliders)
MODEL_ID = os.environ.get("KF_SKY_MODEL", "openmmlab/upernet-convnext-small")


def _load():
    if "m" not in _MODEL:
        import torch
        from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        _MODEL["p"] = AutoImageProcessor.from_pretrained(MODEL_ID)
        _MODEL["m"] = UperNetForSemanticSegmentation.from_pretrained(MODEL_ID).eval()
    return _MODEL["p"], _MODEL["m"]


def _preview(hdr: np.ndarray, size: int = 640) -> np.ndarray:
    Y = luminance(hdr)
    k = 0.25 / max(np.median(Y), 1e-6)
    s = size / min(hdr.shape[:2])
    small = cv2.resize(hdr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) * k
    small = small / (1 + luminance(small)[..., None] / 4.0)  # gentle global tone map
    return (np.clip(srgb_encode(np.clip(small, 0, 1)), 0, 1) * 255).astype(np.uint8)


def _segment(img: np.ndarray) -> np.ndarray:
    """Class probabilities (C, h, w) for a uint8 RGB image."""
    import torch
    proc, model = _load()
    inputs = proc(images=img, return_tensors="pt", do_resize=False)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    logits = torch.nn.functional.interpolate(logits[None], size=img.shape[:2], mode="bilinear",
                                             align_corners=False)[0]
    return torch.softmax(logits, dim=0).numpy().astype(np.float32)


def sky_probability(hdr: np.ndarray, windows: bool = True, size: int = 512) -> np.ndarray:
    """Per-pixel sky probability at preview resolution.

    ADE20K labels sky seen through glass as 'windowpane', so each sizeable window
    region is cropped, enlarged and segmented again - a zoomed-in window view looks
    like an ordinary exterior to the model, which then finds its sky."""
    img = _preview(hdr, size)
    probs = _segment(img)
    prob = probs[SKY_CLASS].copy()
    if windows:
        win = sum(probs[c] for c in WINDOW_CLASSES) > 0.5
        n, lab, stats, _ = cv2.connectedComponentsWithStats(win.astype(np.uint8), 8)
        h, w = win.shape
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < 0.004 * h * w:
                continue
            # crop *inside* the frame: with the frame visible the model keeps
            # calling it a window; the bare view reads as an exterior
            comp = (lab == i).astype(np.uint8)
            k = max(3, int(0.08 * min(bw, bh)))
            inner = cv2.erode(comp, np.ones((k, k), np.uint8))
            ys_, xs_ = np.nonzero(inner if inner.any() else comp)
            x0, x1 = int(xs_.min()), int(xs_.max()) + 1
            y0, y1 = int(ys_.min()), int(ys_.max()) + 1
            if (x1 - x0) < 12 or (y1 - y0) < 12:
                continue
            # re-render the crop from the HDR at higher resolution with its own exposure
            Hh, Wh = hdr.shape[:2]
            sy, sx = Hh / h, Wh / w
            crop = hdr[int(y0 * sy):int(y1 * sy), int(x0 * sx):int(x1 * sx)]
            if min(crop.shape[:2]) < 16:
                continue
            cimg = _preview(crop, size)
            cp = _segment(cimg)[SKY_CLASS]
            cp = cv2.resize(cp, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
            region = (lab[y0:y1, x0:x1] == i)
            region = cv2.dilate(region.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            prob[y0:y1, x0:x1] = np.where(region, np.maximum(prob[y0:y1, x0:x1], cp),
                                          prob[y0:y1, x0:x1])
    return prob


def refine_mask(prob_small: np.ndarray, hdr: np.ndarray) -> np.ndarray:
    """Full-resolution soft sky matte."""
    H, W = hdr.shape[:2]
    prob = cv2.resize(prob_small, (W, H), interpolation=cv2.INTER_LINEAR)
    Y = np.maximum(luminance(hdr), 1e-6)
    L = np.log2(Y).astype(np.float32)
    core_sky = (prob > 0.9).astype(np.float32)
    core_fg = (prob < 0.1).astype(np.float32)
    band = ((prob >= 0.1) & (prob <= 0.9)).astype(np.uint8)
    rb = max(3, int(0.006 * max(H, W)))
    band = cv2.dilate(band, np.ones((rb, rb), np.uint8)).astype(bool)
    # local sky / foreground log-luminance by normalised convolution
    sig = 0.02 * max(H, W)
    def ncov(x, w):
        a = cv2.GaussianBlur(x * w, (0, 0), sig)
        b = cv2.GaussianBlur(w, (0, 0), sig)
        return a / np.maximum(b, 1e-4), b
    Ls, ws = ncov(L, core_sky)
    Lf, wf = ncov(L, core_fg)
    gap = Ls - Lf
    alpha_key = np.clip((L - Lf) / np.maximum(gap, 1e-3), 0, 1)
    usable = (gap > 1.0) & (ws > 0.02) & (wf > 0.02)  # sky clearly brighter than what's in front
    alpha = np.where(prob > 0.5, 1.0, 0.0).astype(np.float32)
    alpha = np.where(band & usable, alpha_key, alpha)
    alpha = np.where(band & ~usable, prob, alpha)
    # light edge-aware cleanup
    alpha = guided_filter(L, alpha.astype(np.float32), 2, 1e-3)
    return np.clip(alpha, 0, 1).astype(np.float32)


def procedural_sky(W: int, H: int, horizon: float, kind: str = "clear") -> np.ndarray:
    """Clean daylight gradient (zenith blue -> pale horizon), sRGB float."""
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None] / max(horizon, 1e-3)
    y = np.clip(y, 0, 1)
    if kind == "dusk":
        top = np.array([0.10, 0.16, 0.36]); mid = np.array([0.36, 0.42, 0.66]); bot = np.array([0.98, 0.70, 0.45])
    else:
        top = np.array([0.22, 0.45, 0.82]); mid = np.array([0.45, 0.66, 0.92]); bot = np.array([0.80, 0.88, 0.96])
    t = y ** 1.6
    col = np.where(t[..., None] < 0.6, top + (mid - top) * (t[..., None] / 0.6),
                   mid + (bot - mid) * ((t[..., None] - 0.6) / 0.4))
    return np.broadcast_to(col, (H, W, 3)).astype(np.float32).copy()


def replace_sky(hdr: np.ndarray, sky: str | np.ndarray | None = None, strength: float = 1.0,
                brightness: float = 1.0, prob: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Replace the sky in a linear HDR image. `sky` is a path to an image, an sRGB
    float array, or None / "clear" / "dusk" for a procedural sky."""
    H, W = hdr.shape[:2]
    if prob is None:
        prob = sky_probability(hdr)
    alpha = refine_mask(prob, hdr) * strength
    info = {"sky_fraction": float(alpha.mean())}
    if alpha.mean() < 0.002:
        info["applied"] = False
        return hdr, info
    rows = np.where(alpha.max(axis=1) > 0.5)[0]
    horizon = (rows.max() + 1) / H if rows.size else 0.5
    info["horizon"] = round(float(horizon), 3)
    if sky is None or isinstance(sky, str) and sky in ("clear", "dusk"):
        src = procedural_sky(W, H, horizon, sky or "clear")
    else:
        img = cv2.imread(sky, cv2.IMREAD_UNCHANGED)[:, :, ::-1] if isinstance(sky, str) else sky
        img = img.astype(np.float32) / (255.0 if img.dtype == np.uint8 else 65535.0 if img.dtype == np.uint16 else 1.0)
        sh, sw = img.shape[:2]
        need_h = int(horizon * H * 1.08) + 1
        sc = max(W / sw, need_h / sh)
        img = cv2.resize(img, (int(np.ceil(sw * sc)), int(np.ceil(sh * sc))), interpolation=cv2.INTER_CUBIC)
        x0 = (img.shape[1] - W) // 2
        y0 = max(0, img.shape[0] - need_h)  # anchor the sky's bottom just below the horizon
        crop = img[y0:y0 + H, x0:x0 + W]
        src = np.zeros((H, W, 3), np.float32)
        src[:crop.shape[0]] = crop
        if crop.shape[0] < H:
            src[crop.shape[0]:] = crop[-1]
    lin = srgb_decode(src).astype(np.float32)
    # light near the horizon is hazier: lift the bottom 12% of the sky slightly
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None] / max(horizon, 1e-3)
    haze = np.clip((yy - 0.88) / 0.12, 0, 1)[..., None]
    lin = lin * (1 - 0.25 * haze) + luminance(lin)[..., None] * 1.15 * (0.25 * haze)
    # exposure match: new sky's bright parts sit where the old sky's were
    sel = alpha > 0.9
    old = luminance(hdr)[sel]
    new = luminance(lin)[sel]
    if old.size > 100:
        k = np.percentile(old, 75) / max(np.percentile(new, 75), 1e-6)
    else:
        k = 1.0
    lin *= k * brightness
    out = hdr * (1 - alpha[..., None]) + lin * alpha[..., None]
    info.update({"applied": True, "exposure_match": round(float(k), 4)})
    return out.astype(np.float32), info
