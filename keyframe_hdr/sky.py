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
    # The window pass is for interiors (sky seen through glass). On exteriors the
    # "windows" it finds are balustrades and facade glass, where it would mistake
    # the distant view (hazy hills, snow) for sky.
    if windows and float((prob > 0.5).mean()) > 0.05:
        windows = False
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


def colour_posterior(prob_small: np.ndarray, hdr: np.ndarray, bins: int = 24) -> np.ndarray:
    """Second opinion on uncertain pixels from this image's own colours.

    The segmentation model can be unsure about plain surfaces next to the sky (a
    soffit, a pale render wall). Colour/brightness histograms are learned from the
    pixels it *is* sure about - this sky versus this house - and every uncertain
    pixel gets the Bayes posterior with the model's probability as prior."""
    h, w = prob_small.shape
    small = cv2.resize(hdr, (w, h), interpolation=cv2.INTER_AREA)
    Y = np.maximum(luminance(small), 1e-7)
    L = np.log2(Y)
    eps = 1e-4 * float(np.median(Y)) + 1e-9
    u = np.log2((small[..., 2] + eps) / (small[..., 1] + eps))  # blue vs green
    v = np.log2((small[..., 0] + eps) / (small[..., 1] + eps))  # red vs green
    lo_L, hi_L = np.percentile(L, 0.5), np.percentile(L, 99.9) + 1e-3
    f = np.stack([np.clip((L - lo_L) / (hi_L - lo_L), 0, 0.999),
                  np.clip((u + 1.5) / 3.0, 0, 0.999), np.clip((v + 1.5) / 3.0, 0, 0.999)], -1)
    idx = (f * bins).astype(np.int32)
    flat = (idx[..., 0] * bins + idx[..., 1]) * bins + idx[..., 2]
    sky_core, fg_core = prob_small > 0.9, prob_small < 0.1
    if sky_core.sum() < 200 or fg_core.sum() < 200:
        return prob_small
    def hist(sel):
        hh = np.bincount(flat[sel], minlength=bins ** 3).astype(np.float32).reshape(bins, bins, bins)
        from scipy.ndimage import gaussian_filter
        hh = gaussian_filter(hh, 1.0)
        return hh / hh.sum()
    ps = hist(sky_core).ravel()[flat] + 1e-7
    pf = hist(fg_core).ravel()[flat] + 1e-7
    pr = np.clip(prob_small, 0.02, 0.98)
    post = pr * ps / (pr * ps + (1 - pr) * pf)
    unsure = (prob_small >= 0.1) & (prob_small <= 0.9)
    unsure = cv2.dilate(unsure.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    out = np.where(unsure, post, prob_small).astype(np.float32)
    return cv2.GaussianBlur(out, (0, 0), 0.7)


def refine_mask(prob_small: np.ndarray, hdr: np.ndarray, colour: bool = True) -> np.ndarray:
    """Full-resolution soft sky matte."""
    H, W = hdr.shape[:2]
    if colour:
        prob_small = colour_posterior(prob_small, hdr)
    prob = cv2.resize(prob_small, (W, H), interpolation=cv2.INTER_LINEAR)
    Y = np.maximum(luminance(hdr), 1e-6)
    L = np.log2(Y).astype(np.float32)
    core_sky = (prob > 0.9).astype(np.float32)
    core_fg = (prob < 0.1).astype(np.float32)
    band = ((prob >= 0.1) & (prob <= 0.9)).astype(np.uint8)
    rb = max(3, int(0.006 * max(H, W)))
    band = cv2.dilate(band, np.ones((rb, rb), np.uint8)).astype(bool)
    # local sky / foreground log-luminance by normalised convolution. These are
    # smooth (sigma = 2% of the frame), so they are computed on a ~1000 px grid and
    # scaled up: same result, minutes faster and far less memory at 45 MP.
    f = min(1.0, 1000.0 / max(H, W))
    hs, wsz = max(1, int(round(H * f))), max(1, int(round(W * f)))
    sig = 0.02 * max(hs, wsz)
    def ncov(x, w):
        xs = cv2.resize(x * w, (wsz, hs), interpolation=cv2.INTER_AREA)
        wsm = cv2.resize(w, (wsz, hs), interpolation=cv2.INTER_AREA)
        a = cv2.GaussianBlur(xs, (0, 0), sig)
        b = cv2.GaussianBlur(wsm, (0, 0), sig)
        up = lambda z: cv2.resize(z, (W, H), interpolation=cv2.INTER_LINEAR)
        return up(a / np.maximum(b, 1e-4)), up(b)
    Ls, ws = ncov(L, core_sky)
    Lf, wf = ncov(L, core_fg)
    gap = Ls - Lf
    # sky only where a pixel is about as bright as the sky around it: a sunlit post
    # top or a pale wall a stop or more darker stays solid; anti-aliased edges and
    # thin branches land on the ramp and mix
    t = np.clip((L - (Ls - 1.6)) / 1.0, 0, 1)
    alpha_key = t * t * (3 - 2 * t)
    # ...and only where the model leans sky at all: bright snow or a white wall on
    # the foreground side of the boundary is as bright as the sky but isn't sky
    alpha_key *= np.clip((prob - 0.05) / 0.25, 0, 1)
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
                brightness: float = 1.0, prob: np.ndarray | None = None,
                chroma: float = 0.5) -> tuple[np.ndarray, dict]:
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
    # Library skies are finished photos; the house grade (vibrance, sky saturation)
    # runs again after compositing, so pull their chroma back first.
    if chroma != 1.0 and not (sky is None or isinstance(sky, str) and sky in ("clear", "dusk")):
        from .grade import oklab_to_rec2020, rec2020_to_oklab
        lab = rec2020_to_oklab(lin)
        # bright cloud tops (often clipped in the source JPEG) go neutral white
        hi = np.clip((lab[..., 0:1] - 0.80) / 0.15, 0, 1)
        lab[..., 1:] *= chroma * (1 - 0.85 * hi)
        lin = np.maximum(oklab_to_rec2020(lab), 0).astype(np.float32)
    # light near the horizon is hazier: lift the bottom 12% of the sky slightly
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None] / max(horizon, 1e-3)
    haze = np.clip((yy - 0.88) / 0.12, 0, 1)[..., None]
    lin = lin * (1 - 0.25 * haze) + luminance(lin)[..., None] * 1.15 * (0.25 * haze)
    # exposure match: new sky's bright parts sit where the old sky's were
    sel = alpha > 0.9
    old = luminance(hdr)[sel]
    new = luminance(lin)[sel]
    if old.size > 100:
        k = np.percentile(old, 50) / max(np.percentile(new, 50), 1e-6)
    else:
        k = 1.0
    lin *= k * brightness
    out = hdr * (1 - alpha[..., None]) + lin * alpha[..., None]
    info.update({"applied": True, "exposure_match": round(float(k), 4)})
    return out.astype(np.float32), info
