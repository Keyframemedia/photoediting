"""Local tone mapping of linear HDR radiance to display-referred luminance.

Luminance-only exposure fusion: the HDR is rendered at several virtual exposures,
each weighted by how well exposed it is, and blended through a Laplacian pyramid
(Mertens et al.). Colour is carried by scaling RGB with the luminance ratio, which
keeps hue and the true relationship between materials intact.
"""
from __future__ import annotations

import cv2
import numexpr as ne
import numpy as np

from .merge import luminance


def srgb_encode(x):
    x = np.asarray(x, dtype=np.float32)
    return ne.evaluate("where(x <= 0.0031308, 12.92 * where(x > 0, x, 0), 1.055 * x ** (1 / 2.4) - 0.055)")


def srgb_decode(v):
    v = np.asarray(v, dtype=np.float32)
    return ne.evaluate("where(v <= 0.04045, where(v > 0, v, 0) / 12.92, ((v + 0.055) / 1.055) ** 2.4)")


def _gauss_pyr(img, levels):
    pyr = [img]
    for _ in range(levels - 1):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def _lap_pyr(img, levels):
    g = _gauss_pyr(img, levels)
    lap = []
    for i in range(levels - 1):
        up = cv2.pyrUp(g[i + 1], dstsize=(g[i].shape[1], g[i].shape[0]))
        lap.append(g[i] - up)
    lap.append(g[-1])
    return lap


def _collapse(lap):
    img = lap[-1]
    for lev in reversed(lap[:-1]):
        img = cv2.pyrUp(img, dstsize=(lev.shape[1], lev.shape[0])) + lev
    return img


def anchored_exposure(Y: np.ndarray, white_target: float = 0.78, white_pct: float = 99.0,
                      window_stops: float = 2.5, median_range=(0.07, 0.22)) -> tuple[float, dict]:
    """Expose like an editor: put the room's brightest real surfaces (walls,
    ceilings, benchtops - not the windows) at `white_target` (linear), then keep the
    scene median within `median_range` so dark materials stay dark and bright
    rooms stay bright."""
    s = Y[:: max(1, Y.shape[0] // 600), :: max(1, Y.shape[1] // 600)].ravel()
    s = np.maximum(s, 1e-7)
    med = float(np.median(s))
    # windows/views: far brighter than the typical surface
    lg = np.log2(s)
    inner = s[lg < np.percentile(lg, 50) + window_stops + 1.5]
    if inner.size < 0.4 * s.size:
        inner = s
    w = float(np.percentile(inner, white_pct))
    k = white_target / max(w, 1e-7)
    lo, hi = median_range
    k = float(np.clip(k, lo / med, hi / med))
    return k, {"median": med, "white": w}


def auto_exposure(Y: np.ndarray, key: float = 0.18, pct: float = 50.0,
                  exclude_top: float = 0.08) -> float:
    """Multiplier that brings the scene's key (median of the non-window pixels)
    to `key`. The brightest `exclude_top` fraction (windows, sky) is ignored."""
    s = Y[:: max(1, Y.shape[0] // 512), :: max(1, Y.shape[1] // 512)].ravel()
    s = s[s > 0]
    hi = np.percentile(s, 100 * (1 - exclude_top))
    s = s[s < hi]
    med = np.exp(np.percentile(np.log(np.maximum(s, 1e-6)), pct))
    return key / max(med, 1e-6)


def _smoothstep(e0, e1, x):
    e0, e1 = np.float32(e0), np.float32(e1)
    return ne.evaluate("where(x <= e0, 0, where(x >= e1, 1, ((x - e0) / (e1 - e0)) ** 2 * (3 - 2 * (x - e0) / (e1 - e0))))")


def fuse_luminance(Y: np.ndarray, evs=(-4.0, -2.5, -1.0, 0.0, 1.0), sigma: float = 0.22,
                   levels: int | None = None, contrast_weight: float = 0.0,
                   center: float = 0.5, mode: str = "hat", ev_prior: float = 0.6,
                   hat=(0.03, 0.14, 0.86, 0.985)) -> np.ndarray:
    """Exposure-fuse virtual exposures of linear luminance Y (already scaled so
    0.18 = mid grey). Returns display-encoded luminance in [0, 1].

    mode="hat": plateau weights with a prior favouring brighter renderings, so each
    region takes the brightest exposure that is not clipped (natural contrast,
    clean whites). mode="gauss": classic Mertens well-exposedness (flatter)."""
    H, W = Y.shape
    if levels is None:
        levels = int(np.floor(np.log2(min(H, W)))) - 3
    acc = None
    wsum = None
    for ev in evs:
        sc = np.float32(2.0 ** ev)
        v = srgb_encode(ne.evaluate("where(Y * sc < 1, Y * sc, 1)"))
        if mode == "hat":
            w = (_smoothstep(hat[0], hat[1], v) * (1 - _smoothstep(hat[2], hat[3], v))).astype(np.float32)
            w *= np.float32(2.0 ** (ev_prior * ev))
        else:
            w = np.exp(-((v - center) ** 2) / (2 * sigma * sigma)).astype(np.float32)
        if contrast_weight > 0:
            lap = np.abs(cv2.Laplacian(v, cv2.CV_32F, ksize=3))
            w *= (1e-3 + lap) ** contrast_weight
        w += 1e-6
        lp = _lap_pyr(v, levels)
        gw = _gauss_pyr(w, levels)
        if acc is None:
            acc = [l * g for l, g in zip(lp, gw)]
            wsum = [g.copy() for g in gw]
        else:
            for i in range(levels):
                acc[i] += lp[i] * gw[i]
                wsum[i] += gw[i]
    out = _collapse([a / s for a, s in zip(acc, wsum)])
    return np.clip(out, 0, 1)


def apply_luminance(rgb: np.ndarray, Y: np.ndarray, Yd_lin: np.ndarray,
                    desat_highlights: float = 0.6, ratio_ref: float | None = None) -> np.ndarray:
    """Scale linear RGB so its luminance becomes Yd_lin, preserving ratios.
    Very bright scene areas (compressed hard) are nudged toward neutral to avoid
    the neon look of ratio-preserving tone mapping."""
    ratio = Yd_lin / np.maximum(Y, 1e-7)
    out = rgb * ratio[:, :, None]
    # where compression is strong (ratio << exposure), mix a touch toward luminance
    if desat_highlights > 0:
        rr = ratio_ref if ratio_ref is not None else np.percentile(ratio, 60)
        comp = np.clip(1 - ratio / rr, 0, 1) * np.clip(Yd_lin / 0.5, 0, 1)
        m = (desat_highlights * comp)[:, :, None]
        out = out * (1 - m * 0.35) + Yd_lin[:, :, None] * (m * 0.35)
    return out
