"""Automatic upright (two-point perspective) correction.

Vertical architectural lines are detected, the 3-D vertical direction is
estimated with the real focal length from EXIF, and the image is re-projected as
if the camera had been perfectly level (pitch and roll removed, yaw kept). The
result is cropped to the largest clean rectangle at the original aspect ratio,
without upsampling.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .merge import luminance


def _intrinsics(meta: dict, W: int, H: int) -> np.ndarray:
    f35 = float(meta.get("FocalLengthIn35mmFormat") or 0) or float(meta.get("FocalLength") or 24)
    f = f35 / 36.0 * max(W, H)
    return np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1]], dtype=np.float64)


def detect_verticals(Y: np.ndarray, max_tilt_deg: float = 25.0, min_len_frac: float = 0.04):
    """Near-vertical line segments (x1, y1, x2, y2) on a ~1600px image."""
    L = np.log2(np.maximum(Y, 1e-5))
    L = cv2.normalize(L, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    segs = lsd.detect(L)[0]
    if segs is None:
        return np.zeros((0, 4))
    segs = segs.reshape(-1, 4)
    dx = segs[:, 2] - segs[:, 0]
    dy = segs[:, 3] - segs[:, 1]
    length = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(np.abs(dx), np.abs(dy)))  # 0 = vertical
    keep = (ang < max_tilt_deg) & (length > min_len_frac * Y.shape[0])
    return segs[keep]


def vertical_direction(segs: np.ndarray, K: np.ndarray, iters: int = 400, seed: int = 0):
    """Robustly estimate the 3-D vertical direction (camera coords) from segments.
    Each segment defines an interpretation plane; the vertical lies in all of them."""
    if len(segs) < 4:
        return None, 0
    p1 = np.c_[segs[:, :2], np.ones(len(segs))]
    p2 = np.c_[segs[:, 2:], np.ones(len(segs))]
    lines = np.cross(p1, p2)
    n = (K.T @ lines.T).T
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    w = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    rng = np.random.default_rng(seed)
    best, best_score, best_inl = None, -1, None
    thr = math.sin(math.radians(0.6))
    for _ in range(iters):
        i, j = rng.choice(len(segs), 2, replace=False, p=w / w.sum())
        d = np.cross(n[i], n[j])
        nd = np.linalg.norm(d)
        if nd < 1e-6:
            continue
        d /= nd
        if d[1] < 0:
            d = -d
        if d[1] < math.cos(math.radians(30)):  # must be near the image vertical
            continue
        res = np.abs(n @ d)
        inl = res < thr
        score = w[inl].sum()
        if score > best_score:
            best, best_score, best_inl = d, score, inl
    if best is None:
        return None, 0
    # weighted least-squares refinement on inliers
    A = (n[best_inl] * w[best_inl, None]).T @ n[best_inl]
    evals, evecs = np.linalg.eigh(A)
    d = evecs[:, 0]
    if d[1] < 0:
        d = -d
    return d, int(best_inl.sum())


def _rotation_to_y(d: np.ndarray) -> np.ndarray:
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(d, y)
    s = np.linalg.norm(v)
    c = float(np.dot(d, y))
    if s < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def _inscribed_rect(poly: np.ndarray, W: int, H: int):
    """Largest rectangle with aspect W/H inside the convex polygon."""
    poly = poly.astype(np.float32).reshape(-1, 1, 2)
    best = (0, 0, 0)
    cx0, cy0 = (W - 1) / 2, (H - 1) / 2
    for ox in np.linspace(-0.04, 0.04, 5) * W:
        for oy in np.linspace(-0.06, 0.06, 7) * H:
            cx, cy = cx0 + ox, cy0 + oy
            lo, hi = 0.0, 1.0
            for _ in range(22):
                s = (lo + hi) / 2
                hw, hh = s * W / 2, s * H / 2
                pts = [(cx - hw, cy - hh), (cx + hw, cy - hh), (cx - hw, cy + hh), (cx + hw, cy + hh)]
                if all(cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0 for x, y in pts):
                    lo = s
                else:
                    hi = s
            if lo > best[0]:
                best = (lo, cx, cy)
    return best


def upright(hdr: np.ndarray, meta: dict, p: dict) -> tuple[np.ndarray, dict]:
    H, W = hdr.shape[:2]
    s = 1600 / max(H, W)
    Y = cv2.resize(luminance(hdr), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    segs = detect_verticals(Y)
    Ks = _intrinsics(meta, Y.shape[1], Y.shape[0])
    d, ninl = vertical_direction(segs, Ks)
    info = {"segments": int(len(segs)), "inliers": ninl}
    if d is None or ninl < p.get("upright_min_inliers", 6):
        info["applied"] = False
        return hdr, info
    R = _rotation_to_y(d)
    ang = math.degrees(math.acos(min(1.0, float(d[1]))))
    pitch = math.degrees(math.atan2(d[2], d[1]))
    roll = math.degrees(math.atan2(d[0], d[1]))
    info.update({"tilt_deg": round(ang, 2), "pitch_deg": round(pitch, 2), "roll_deg": round(roll, 2)})
    if ang < 0.15 or ang > p.get("upright_max_deg", 14.0):
        info["applied"] = False
        return hdr, info
    strength = p.get("upright_strength", 1.0)
    if strength < 1.0:
        axis_angle = cv2.Rodrigues(R)[0] * strength
        R = cv2.Rodrigues(axis_angle)[0]
    K = _intrinsics(meta, W, H)
    Hm = K @ R @ np.linalg.inv(K)
    corners = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float64)
    wc = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), Hm).reshape(-1, 2)
    sc, cx, cy = _inscribed_rect(wc, W, H)
    if sc < 0.70:
        info["applied"] = False
        info["reason"] = f"crop too large ({sc:.2f})"
        return hdr, info
    ow, oh = int(round(sc * W)), int(round(sc * H))
    T = np.array([[1, 0, -(cx - (ow - 1) / 2)], [0, 1, -(cy - (oh - 1) / 2)], [0, 0, 1]], dtype=np.float64)
    out = cv2.warpPerspective(hdr, T @ Hm, (ow, oh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    info.update({"applied": True, "crop_scale": round(sc, 3), "size": [ow, oh]})
    return np.maximum(out, 0), info


def level(hdr: np.ndarray, meta: dict, p: dict) -> tuple[np.ndarray, dict]:
    """Aerials: the gimbal keeps the horizon level; nothing to do by default."""
    return hdr, {"applied": False, "reason": "aerial"}
