"""Noise reduction and defringing on the merged linear HDR."""
from __future__ import annotations

import cv2
import numpy as np


def guided_filter(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """He et al. guided filter (grey guide), box filters via OpenCV."""
    mean = lambda x: cv2.boxFilter(x, cv2.CV_32F, (2 * r + 1, 2 * r + 1))
    mI, mp = mean(I), mean(p)
    cov = mean(I * p) - mI * mp
    var = mean(I * I) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return mean(a) * I + mean(b)


def in_strips(fn, rgb: np.ndarray, halo: int, rows: int = 1024) -> np.ndarray:
    """Apply a local filter `fn` to `rgb` in overlapping row strips, writing the
    result back in place. Each strip carries `halo` extra rows of original data
    on both sides, so the output equals a whole-image call while peak memory
    stays at one strip's worth of temporaries (45 MP frames run two at a time)."""
    H = rgb.shape[0]
    if H <= rows + 2 * halo:
        rgb[...] = fn(rgb)
        return rgb
    saved = None  # original rows just above the current strip (already overwritten)
    for y0 in range(0, H, rows):
        y1 = min(H, y0 + rows)
        hi = min(H, y1 + halo)
        block = rgb[0:hi] if y0 == 0 else np.concatenate([saved, rgb[y0:hi]], axis=0)
        top = 0 if y0 == 0 else saved.shape[0]
        res = fn(np.ascontiguousarray(block))
        saved = rgb[max(0, y1 - halo):y1].copy()
        rgb[y0:y1] = res[top:top + (y1 - y0)]
    return rgb


def _radius_chroma(shape, strength):
    return max(2, int(round(3 * strength * max(shape[:2]) / 4000)))


def _radius_luma(shape):
    return max(1, int(round(2 * max(shape[:2]) / 8000)))


def clean(rgb: np.ndarray, chroma: float = 1.0, luma: float = 0.35, fringe: float = 1.0) -> np.ndarray:
    """Chroma NR, luma NR and defringe, strip by strip and in place."""
    rc = _radius_chroma(rgb.shape, chroma) if chroma > 0 else 0
    rl = _radius_luma(rgb.shape) if luma > 0 else 0
    halo = 2 * rc + 2 * rl + 4

    def fn(block):
        if chroma > 0:
            block = chroma_nr(block, chroma, r=rc)
        if luma > 0:
            block = luma_nr(block, luma, r=rl)
        if fringe > 0:
            block = defringe(block, fringe)
        return block

    return in_strips(fn, rgb, halo)


def chroma_nr(rgb: np.ndarray, strength: float = 1.0, r: int | None = None) -> np.ndarray:
    """Remove colour noise: smooth chromaticity ratios with an edge-aware filter
    guided by log luminance. Luminance (detail) is untouched."""
    if strength <= 0:
        return rgb
    Y = rgb @ np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)
    Ys = np.maximum(Y, 1e-6)
    guide = np.log2(Ys).astype(np.float32)
    if r is None:
        r = _radius_chroma(rgb.shape, strength)
    out = np.empty_like(rgb)
    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    for c in range(3):
        ratio = np.clip(rgb[..., c] / Ys, 0, 4).astype(np.float32)
        f = guided_filter(guide, ratio, r, 0.02)
        # A guided filter fits a linear model per window and can extrapolate far
        # outside the data at extreme-contrast edges (a sunlit frame against deep
        # shade), which shows as box-shaped neon patches. Keep every output inside
        # the range of chromaticities actually present in its window.
        np.clip(f, cv2.erode(ratio, k), cv2.dilate(ratio, k), out=f)
        # and never move a pixel's chromaticity far from its own value
        np.clip(f, ratio - 0.3, ratio + 0.3, out=f)
        out[..., c] = f
    # chroma NR must not change brightness: renormalise to the original luminance
    lum = out @ np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)
    out *= (Ys / np.maximum(lum, 1e-6))[..., None]
    return np.maximum(out, 0)


def luma_nr(rgb: np.ndarray, amount: float = 0.5, r: int | None = None) -> np.ndarray:
    """Gentle luminance NR in the log domain (noise is roughly multiplicative
    after merging). Edges are preserved by the guided filter."""
    if amount <= 0:
        return rgb
    Y = np.maximum(rgb @ np.array([0.2627, 0.6780, 0.0593], dtype=np.float32), 1e-6)
    L = np.log2(Y).astype(np.float32)
    if r is None:
        r = _radius_luma(rgb.shape)
    Ls = guided_filter(L, L, r, 0.004)
    L2 = L + amount * (Ls - L)
    return rgb * (np.exp2(L2 - L))[..., None]


def defringe(rgb: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Suppress purple/blue-green fringes on high-contrast edges (longitudinal CA)
    by pulling chroma toward the neighbourhood's on edge pixels with fringe hues."""
    if amount <= 0:
        return rgb
    Y = np.maximum(rgb @ np.array([0.2627, 0.6780, 0.0593], dtype=np.float32), 1e-6)
    L = np.log2(Y)
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.clip((np.hypot(gx, gy) - 1.0) / 2.0, 0, 1)
    edge = cv2.dilate(edge, np.ones((3, 3), np.uint8))
    r, g, b = rgb[..., 0] / Y, rgb[..., 1] / Y, rgb[..., 2] / Y
    # purple/magenta: r and b high vs g; cyan/blue fringe: b high vs r
    purple = np.clip((np.minimum(r, b) - g) / 0.15, 0, 1)
    out = rgb.copy()
    m = (edge * purple * amount)[..., None]
    out = out * (1 - m) + Y[..., None] * m
    return out
