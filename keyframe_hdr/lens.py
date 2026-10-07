"""Lens corrections: DNG opcodes (WarpRectilinear, GainMap), Adobe LCP vignetting
and Nikon embedded vignette coefficients.

Distortion and lateral CA come from the DNG's own WarpRectilinear opcode (Adobe DNG
Converter writes it from its built-in profile for Nikon Z and Tamron lenses; DJI
writes its own). Vignetting is not in the converted Nikon DNGs, so it is taken from
an Adobe LCP profile when one exists for the lens, otherwise from the Nikon maker
note coefficients.
"""
from __future__ import annotations

import glob
import math
import os
import re
import struct
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import cv2
import numpy as np

LCP_ROOT_CANDIDATES = [
    os.environ.get("KF_LCP_ROOT", ""),
    "/root/.wine/drive_c/ProgramData/Adobe/CameraRaw/LensProfiles/1.0",
]

# ---------------------------------------------------------------------------
# DNG opcodes
# ---------------------------------------------------------------------------


@dataclass
class WarpRectilinear:
    planes: list  # per plane (kr0, kr1, kr2, kr3, kt0, kt1)
    cx: float
    cy: float


@dataclass
class GainMap:
    top: int
    left: int
    bottom: int
    right: int
    plane: int
    planes: int
    points_v: int
    points_h: int
    spacing_v: float
    spacing_h: float
    origin_v: float
    origin_h: float
    map_planes: int
    gains: np.ndarray  # (points_v, points_h, map_planes)


@dataclass
class Opcodes:
    warp: WarpRectilinear | None = None
    gainmaps: list = field(default_factory=list)


def read_opcode_list3(path: str) -> Opcodes:
    blob = subprocess.run(
        ["exiftool", "-b", "-OpcodeList3", path], capture_output=True, check=False
    ).stdout
    ops = Opcodes()
    if len(blob) < 4:
        return ops
    n = struct.unpack(">I", blob[:4])[0]
    p = 4
    for _ in range(n):
        oid, _ver, _flags, nb = struct.unpack(">IIII", blob[p : p + 16])
        p += 16
        prm = blob[p : p + nb]
        p += nb
        if oid == 1:  # WarpRectilinear
            planes = struct.unpack(">I", prm[:4])[0]
            vals = struct.unpack(">" + "d" * (planes * 6 + 2), prm[4 : 4 + 8 * (planes * 6 + 2)])
            ops.warp = WarpRectilinear(
                [vals[i * 6 : i * 6 + 6] for i in range(planes)], vals[-2], vals[-1]
            )
        elif oid == 9:  # GainMap
            t, l, b, r, pl, pls, _rp, _cp, mpv, mph = struct.unpack(">10I", prm[:40])
            sv, sh, ov, oh = struct.unpack(">4d", prm[40:72])
            mpl = struct.unpack(">I", prm[72:76])[0]
            g = np.frombuffer(prm[76 : 76 + 4 * mpv * mph * mpl], dtype=">f4")
            ops.gainmaps.append(
                GainMap(t, l, b, r, pl, pls, mpv, mph, sv, sh, ov, oh, mpl,
                        g.reshape(mpv, mph, mpl).astype(np.float32))
            )
    return ops


def apply_gainmaps(img: np.ndarray, gainmaps: list, scale: float = 1.0) -> np.ndarray:
    """Apply DNG GainMap opcodes in place. `scale` = output pixels per stage-3 pixel
    (0.5 for half-size decodes)."""
    H, W = img.shape[:2]
    for gm in gainmaps:
        t, l = int(gm.top * scale), int(gm.left * scale)
        b, r = int(gm.bottom * scale), int(gm.right * scale)
        b, r = min(b, H), min(r, W)
        hh, ww = b - t, r - l
        # normalised coordinates of pixel centres within the area
        v = (np.arange(hh, dtype=np.float32) + 0.5) / hh
        u = (np.arange(ww, dtype=np.float32) + 0.5) / ww
        fy = np.clip((v - gm.origin_v) / gm.spacing_v, 0, gm.points_v - 1)
        fx = np.clip((u - gm.origin_h) / gm.spacing_h, 0, gm.points_h - 1)
        mapx = np.broadcast_to(fx[None, :], (hh, ww)).astype(np.float32)
        mapy = np.broadcast_to(fy[:, None], (hh, ww)).astype(np.float32)
        for k in range(gm.planes):
            plane = gm.plane + k
            if plane >= img.shape[2]:
                continue
            mp = min(k, gm.map_planes - 1)
            g = cv2.remap(gm.gains[:, :, mp], mapx, mapy, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
            img[t:b, l:r, plane] *= g
    return img


def warp_maps(warp: WarpRectilinear, H: int, W: int, plane: int, step: int = 8):
    """Source sampling maps for one plane of a WarpRectilinear opcode.

    The warp is a low-order polynomial, so it is evaluated exactly on a coarse
    grid (every `step` px) and bilinearly interpolated - sub-0.01px error, ~50x
    faster than evaluating all 45M pixels."""
    kr0, kr1, kr2, kr3, kt0, kt1 = warp.planes[min(plane, len(warp.planes) - 1)]
    cx = warp.cx * (W - 1)
    cy = warp.cy * (H - 1)
    m = max(math.hypot(cx, cy), math.hypot(W - 1 - cx, cy),
            math.hypot(cx, H - 1 - cy), math.hypot(W - 1 - cx, H - 1 - cy))
    gw, gh = (W - 1) // step + 2, (H - 1) // step + 2
    xs = (np.arange(gw, dtype=np.float64) * step - cx) / m
    ys = (np.arange(gh, dtype=np.float64) * step - cy) / m
    dx, dy = np.meshgrid(xs, ys)
    r2 = dx * dx + dy * dy
    f = kr0 + r2 * (kr1 + r2 * (kr2 + r2 * kr3))
    sx = cx + m * (f * dx + kt0 * 2 * dx * dy + kt1 * (r2 + 2 * dx * dx))
    sy = cy + m * (f * dy + kt1 * 2 * dx * dy + kt0 * (r2 + 2 * dy * dy))
    # interpolate the coarse grid to full resolution
    gx = (np.arange(W, dtype=np.float32) / step)
    gy = (np.arange(H, dtype=np.float32) / step)
    mx = np.broadcast_to(gx[None, :], (H, W)).astype(np.float32)
    my = np.broadcast_to(gy[:, None], (H, W)).astype(np.float32)
    full_x = cv2.remap(sx.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    full_y = cv2.remap(sy.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return full_x, full_y


def apply_warp(img: np.ndarray, warp: WarpRectilinear, mask: np.ndarray | None = None):
    """Warp each plane with its own map (lateral CA); the mask follows green."""
    H, W = img.shape[:2]
    out = np.empty_like(img)
    cache, green = None, None
    for c in range(img.shape[2]):
        same = c > 0 and warp.planes[min(c, len(warp.planes) - 1)] == warp.planes[min(c - 1, len(warp.planes) - 1)]
        if not same or cache is None:
            cache = warp_maps(warp, H, W, c)
        if c == 1:
            green = cache
        out[:, :, c] = cv2.remap(img[:, :, c], cache[0], cache[1], cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REFLECT)
    if mask is not None:
        g = green or cache
        mask = cv2.remap(mask, g[0], g[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return out, mask
    return out


# ---------------------------------------------------------------------------
# Vignetting
# ---------------------------------------------------------------------------


def _radius_grid(H: int, W: int):
    ys = np.arange(H, dtype=np.float32) - (H - 1) / 2
    xs = np.arange(W, dtype=np.float32) - (W - 1) / 2
    return np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2)


def nikon_vignette_gain(coeffs, H: int, W: int, max_gain: float = 3.5):
    """Gain map from Nikon maker-note VignetteCoefficient1..3.

    Model: gain = 1 + v1 r^2 + v2 r^4 + v3 r^6, r normalised to the half diagonal.
    Returns None when the coefficients do not describe a plausible correction
    (gain < 1 anywhere or extreme), which happens for some third-party lenses.
    """
    v1, v2, v3 = coeffs
    rr = np.linspace(0, 1, 64)
    g = 1 + v1 * rr**2 + v2 * rr**4 + v3 * rr**6
    if g.min() < 0.98 or g.max() > max_gain or np.any(np.diff(g) < -1e-3):
        return None
    hs, ws = max(2, H // 8), max(2, W // 8)
    r = _radius_grid(hs, ws) / math.hypot((ws - 1) / 2, (hs - 1) / 2)
    r2 = r * r
    g = (1 + r2 * (v1 + r2 * (v2 + r2 * v3))).astype(np.float32)
    return cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)


@dataclass
class LcpEntry:
    focal: float
    distance: float
    aperture_value: float
    vig: tuple | None
    piecewise: list | None
    fx: float | None


_LCP_CACHE: dict = {}


def find_lcp(lens_model: str) -> str | None:
    if not lens_model:
        return None
    root = next((r for r in LCP_ROOT_CANDIDATES if r and os.path.isdir(r)), None)
    if root is None:
        return None
    # Match on the distinctive model code (e.g. A058) or the focal/aperture string.
    tokens = re.findall(r"[A-Z]\d{3}|\d+-\d+mm|\d+mm", lens_model.replace(" ", " "))
    code = next((t for t in tokens if re.fullmatch(r"[A-Z]\d{3}", t)), None)
    maker = lens_model.split()[0].capitalize()
    cands = glob.glob(os.path.join(root, "*", "**", "*RAW.lcp"), recursive=True)
    best = None
    for c in cands:
        name = os.path.basename(c)
        if code and code in name and maker.lower() in c.lower():
            best = c
            break
    return best


def parse_lcp(path: str) -> list:
    if path in _LCP_CACHE:
        return _LCP_CACHE[path]
    ns = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "st": "http://ns.adobe.com/photoshop/1.0/camera-profile",
    }
    tree = ET.parse(path)
    out = []
    S = "{%s}" % ns["st"]
    for d in tree.iter("{%s}Description" % ns["rdf"]):
        if d.get(S + "FocalLength") is None:
            continue
        pm = d.find("st:PerspectiveModel/rdf:Description", ns)
        if pm is None:
            pm = d.find("st:PerspectiveModel", ns)
        if pm is None:
            continue
        # The vignette model is either an element carrying attributes or an element
        # wrapping an rdf:Description; FocalLengthX may sit on either level.
        vm = pm.find("st:VignetteModel", ns)
        vig, pw, fx = None, None, pm.get(S + "FocalLengthX")
        if vm is not None:
            inner = vm.find("rdf:Description", ns)
            src = inner if inner is not None and inner.get(S + "VignetteModelParam1") else vm
            if src.get(S + "VignetteModelParam1") is not None:
                vig = tuple(float(src.get(S + f"VignetteModelParam{i}", 0)) for i in (1, 2, 3))
            fx = src.get(S + "FocalLengthX") or fx
            seq = src.find("st:VignetteModelPiecewiseParam/rdf:Seq", ns)
            if seq is not None:
                pw = [tuple(float(x) for x in li.text.split(",")) for li in seq]
        out.append(LcpEntry(
            float(d.get(S + "FocalLength")),
            float(d.get(S + "FocusDistance", 10000)),
            float(d.get(S + "ApertureValue", 0)),
            vig, pw, float(fx) if fx else None,
        ))
    _LCP_CACHE[path] = out
    return out


def _lcp_v(entry: LcpEntry, r: np.ndarray) -> np.ndarray:
    if entry.piecewise:
        pr = np.array([p[0] for p in entry.piecewise])
        pv = np.array([p[1] for p in entry.piecewise])
        return np.interp(r, pr, pv, right=pv[-1])
    a1, a2, a3 = entry.vig
    r2 = r * r
    return 1 + r2 * (a1 + r2 * (a2 + r2 * a3))


def lcp_vignette_gain(path: str, focal: float, fnumber: float, distance: float,
                      H: int, W: int, sensor_long_mm: float = 36.0):
    entries = [e for e in parse_lcp(path) if e.vig is not None or e.piecewise]
    if not entries:
        return None
    av = 2 * math.log2(max(fnumber, 1.0))
    # closest aperture value, then closest distance (in log space)
    best_av = min({e.aperture_value for e in entries}, key=lambda a: abs(a - av))
    sel = [e for e in entries if e.aperture_value == best_av]
    ld = math.log(max(distance, 0.1))
    best_d = min({e.distance for e in sel}, key=lambda d: abs(math.log(max(d, 0.1)) - ld))
    sel = sorted([e for e in sel if e.distance == best_d], key=lambda e: e.focal)
    lo = max([e for e in sel if e.focal <= focal], key=lambda e: e.focal, default=sel[0])
    hi = min([e for e in sel if e.focal >= focal], key=lambda e: e.focal, default=sel[-1])
    t = 0.0 if hi.focal == lo.focal else (focal - lo.focal) / (hi.focal - lo.focal)
    dmax = max(H, W)
    hs, ws = max(2, H // 8), max(2, W // 8)
    rad = _radius_grid(hs, ws) * (W / ws)
    rs = np.linspace(0, rad.max(), 512)

    def curve(e):
        fnorm = e.fx if e.fx else e.focal / sensor_long_mm
        return _lcp_v(e, rs / (fnorm * dmax))

    v = (1 - t) * curve(lo) + t * curve(hi)
    v = np.clip(v, 0.15, 1.0)
    g = np.interp(rad, rs, 1.0 / v).astype(np.float32)
    return cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
