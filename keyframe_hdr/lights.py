"""Twilight light enhancement: every interior and exterior light reads as on.

Emitters (lamps, downlights, wall lights, lit windows) are found in the linear
HDR, where a light is simply much brighter than its surroundings. After the
grade, they get a soft multi-scale glow in their own colour and the pools of
light they throw are lifted a little, the way the eye sees a lit house at dusk.
Skies are excluded, so a bright horizon never "glows".

NOTE: tuned without a real twilight set; strengths live in presets.NIGHT.
"""
from __future__ import annotations

import cv2
import numpy as np

from .merge import luminance
from .tonemap import srgb_decode, srgb_encode


def _smooth(lo, hi, x):
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def emitter_map(hdr: np.ndarray, sky: np.ndarray | None = None, long_edge: int = 1600) -> np.ndarray:
    """Soft map (reduced size) of light sources and lit windows, 0..1."""
    H, W = hdr.shape[:2]
    s = min(1.0, long_edge / max(H, W))
    small = cv2.resize(hdr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    Y = np.maximum(luminance(small), 1e-7)
    L = np.log2(Y)
    lk = float(np.log2(np.median(Y)))
    sig = 0.015 * max(small.shape[:2])
    Lb = np.log2(np.maximum(cv2.GaussianBlur(Y, (0, 0), sig), 1e-7))
    # much brighter than the neighbourhood, and bright in absolute terms
    m = _smooth(1.0, 2.6, L - Lb) * _smooth(lk + 1.5, lk + 3.5, L)
    # lit windows: large areas well above the scene key, not as peaky as bulbs
    Lb2 = np.log2(np.maximum(cv2.GaussianBlur(Y, (0, 0), 4 * sig), 1e-7))
    win = _smooth(0.6, 1.8, L - Lb2) * _smooth(lk + 0.8, lk + 2.5, L)
    m = np.maximum(m, 0.6 * win)
    if sky is not None:
        sk = cv2.resize(sky.astype(np.float32), (m.shape[1], m.shape[0]), interpolation=cv2.INTER_AREA)
        m *= 1 - np.clip(sk * 1.5, 0, 1)
    return m.astype(np.float32)


def enhance(v: np.ndarray, emit: np.ndarray, glow: float = 0.35, pool: float = 0.18,
            warmth: float = 0.04) -> np.ndarray:
    """Add glow and lift light pools on a finished sRGB image (in place)."""
    if glow <= 0 and pool <= 0:
        return v
    H, W = v.shape[:2]
    h, w = emit.shape
    small = cv2.resize(v, (w, h), interpolation=cv2.INTER_AREA)
    lin = srgb_decode(np.clip(small, 0, 1)).astype(np.float32)
    src = lin * emit[..., None]
    if warmth:
        src *= np.array([1 + warmth, 1.0, 1 - 2 * warmth], np.float32)
    n = max(h, w)
    bloom = (0.50 * cv2.GaussianBlur(src, (0, 0), 0.003 * n)
             + 0.35 * cv2.GaussianBlur(src, (0, 0), 0.010 * n)
             + 0.25 * cv2.GaussianBlur(src, (0, 0), 0.030 * n))
    lift = cv2.GaussianBlur(emit, (0, 0), 0.04 * n)
    lift = lift / max(float(lift.max()), 1e-6)
    for r0 in range(0, H, 512):
        r1 = min(H, r0 + 512)
        b = _resize_rows(bloom, H, W, r0, r1)
        l = _resize_rows(lift[..., None], H, W, r0, r1)[..., 0]
        x = srgb_decode(np.clip(v[r0:r1], 0, 1)).astype(np.float32)
        x *= (1 + pool * l)[..., None]
        x = 1 - (1 - np.clip(x, 0, 1)) * (1 - np.clip(glow * b, 0, 1))  # screen
        v[r0:r1] = srgb_encode(np.clip(x, 0, 1))
    return v


def _resize_rows(img: np.ndarray, H: int, W: int, r0: int, r1: int) -> np.ndarray:
    """Rows r0..r1 of `img` bilinearly resized to H x W, without building the
    full-size image."""
    h, w = img.shape[:2]
    ys = ((np.arange(r0, r1) + 0.5) * h / H - 0.5).astype(np.float32)
    xs = ((np.arange(W) + 0.5) * w / W - 0.5).astype(np.float32)
    mx, my = np.meshgrid(xs, ys)
    return cv2.remap(img.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(r1 - r0, W, -1)
