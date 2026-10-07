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


def chroma_nr(rgb: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Remove colour noise: smooth chromaticity ratios with an edge-aware filter
    guided by log luminance. Luminance (detail) is untouched."""
    if strength <= 0:
        return rgb
    Y = rgb @ np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)
    Ys = np.maximum(Y, 1e-6)
    guide = np.log2(Ys).astype(np.float32)
    r = max(2, int(round(3 * strength * max(rgb.shape[:2]) / 4000)))
    out = np.empty_like(rgb)
    for c in range(3):
        ratio = (rgb[..., c] / Ys).astype(np.float32)
        out[..., c] = guided_filter(guide, ratio, r, 0.02) * Ys
    return np.maximum(out, 0)


def luma_nr(rgb: np.ndarray, amount: float = 0.5) -> np.ndarray:
    """Gentle luminance NR in the log domain (noise is roughly multiplicative
    after merging). Edges are preserved by the guided filter."""
    if amount <= 0:
        return rgb
    Y = np.maximum(rgb @ np.array([0.2627, 0.6780, 0.0593], dtype=np.float32), 1e-6)
    L = np.log2(Y).astype(np.float32)
    r = max(1, int(round(2 * max(rgb.shape[:2]) / 8000)))
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
