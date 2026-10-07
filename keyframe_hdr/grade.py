"""Signature colour grade: white balance, tone curve, clarity, colour, sharpening.

All colour work happens in linear Rec.2020 or Oklab (perceptual) so hues stay true;
the result is gamut-mapped to sRGB at the end.
"""
from __future__ import annotations

import cv2
import numexpr as ne
import numpy as np

from .merge import luminance
from .tonemap import srgb_decode, srgb_encode

REC2020_TO_XYZ = np.array([
    [0.6369580, 0.1446169, 0.1688810],
    [0.2627002, 0.6779981, 0.0593017],
    [0.0000000, 0.0280727, 1.0609851],
], dtype=np.float32)
XYZ_TO_REC2020 = np.linalg.inv(REC2020_TO_XYZ).astype(np.float32)
XYZ_TO_SRGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
], dtype=np.float32)
REC2020_TO_SRGB = (XYZ_TO_SRGB @ REC2020_TO_XYZ).astype(np.float32)

OK_M1 = np.array([
    [0.8189330101, 0.3618667424, -0.1288597137],
    [0.0329845436, 0.9293118715, 0.0361456387],
    [0.0482003018, 0.2643662691, 0.6338517070],
], dtype=np.float32)
OK_M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
], dtype=np.float32)
OK_M1_INV = np.linalg.inv(OK_M1).astype(np.float32)
OK_M2_INV = np.linalg.inv(OK_M2).astype(np.float32)

BRADFORD = np.array([
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
], dtype=np.float32)


def _mat(img, M):
    return (img.reshape(-1, 3) @ M.T).reshape(img.shape)


def rec2020_to_oklab(rgb):
    lms = _mat(rgb, (OK_M1 @ REC2020_TO_XYZ).astype(np.float32))
    lms = np.cbrt(np.maximum(lms, 0))
    return _mat(lms, OK_M2)


def oklab_to_rec2020(lab):
    lms = _mat(lab, OK_M2_INV)
    lms = lms * lms * lms
    return _mat(lms, (XYZ_TO_REC2020 @ OK_M1_INV).astype(np.float32))


# ---------------------------------------------------------------------------
# White balance
# ---------------------------------------------------------------------------

def _xy_to_xyz(x, y):
    return np.array([x / y, 1.0, (1 - x - y) / y], dtype=np.float32)


def cct_to_xy(cct: float):
    """Planckian locus (Kim et al. cubic spline, 1667K-25000K). Targeting the
    black-body line rather than the daylight locus keeps warm whites clean instead
    of drifting toward olive."""
    t = cct
    if t <= 4000:
        x = -0.2661239e9 / t**3 - 0.2343589e6 / t**2 + 0.8776956e3 / t + 0.179910
    else:
        x = -3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390
    if t <= 2222:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif t <= 4000:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    return x, y


def adapt_matrix(src_xyz: np.ndarray, dst_xyz: np.ndarray) -> np.ndarray:
    """Bradford CAT expressed in Rec.2020 linear."""
    s = BRADFORD @ src_xyz
    d = BRADFORD @ dst_xyz
    M = np.linalg.inv(BRADFORD) @ np.diag(d / s) @ BRADFORD
    return (XYZ_TO_REC2020 @ M @ REC2020_TO_XYZ).astype(np.float32)


def estimate_neutral(rgb: np.ndarray, exposure: float) -> np.ndarray | None:
    """Estimate the colour of light on neutral surfaces (white walls, ceilings,
    benchtops): bright, low-chroma pixels that are not windows or clipped."""
    step = max(1, int(np.sqrt(rgb.shape[0] * rgb.shape[1] / 1.5e6)))
    s = rgb[::step, ::step].reshape(-1, 3) * exposure
    Y = s @ np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)
    lab = rec2020_to_oklab(s.reshape(-1, 1, 3)).reshape(-1, 3)
    C = np.hypot(lab[:, 1], lab[:, 2])
    sel = (Y > 0.08) & (Y < 0.85) & (C < 0.05)
    if sel.sum() < 500:
        return None
    # brightest half of the candidates: walls/ceilings rather than grey floors
    ysel = Y[sel]
    sel2 = sel.copy()
    sel2[sel] = ysel > np.percentile(ysel, 50)
    xyz = s[sel2] @ REC2020_TO_XYZ.T
    m = np.median(xyz / np.maximum(xyz[:, 1:2], 1e-6), axis=0)
    return m.astype(np.float32)


def white_balance(rgb: np.ndarray, exposure: float, target_cct: float, strength: float,
                  tint: float = 0.0, max_shift_mired: float = 45.0,
                  max_warm_mired: float = 12.0, fixed_cct: float | None = None) -> tuple[np.ndarray, dict]:
    """Adapt measured neutral toward a (warm) target white. strength 0 = camera WB.

    fixed_cct: skip estimation and treat the scene illuminant as this colour
    temperature (night/twilight: keeps tungsten glowing warm and dusk skies blue)."""
    D65 = _xy_to_xyz(0.31271, 0.32902)
    tx, ty = cct_to_xy(target_cct)
    ty += tint * 0.01
    target = _xy_to_xyz(tx, ty)
    M, info = wb_matrix(rgb, exposure, target_cct, strength, tint, max_shift_mired,
                        max_warm_mired, fixed_cct)
    return np.maximum(_mat(rgb, M), 0), info


def wb_matrix(rgb: np.ndarray, exposure: float, target_cct: float, strength: float,
              tint: float = 0.0, max_shift_mired: float = 45.0,
              max_warm_mired: float = 12.0, fixed_cct: float | None = None) -> tuple[np.ndarray, dict]:
    """The 3x3 (Rec.2020 linear) white-balance matrix; see white_balance()."""
    D65 = _xy_to_xyz(0.31271, 0.32902)
    tx, ty = cct_to_xy(target_cct)
    ty += tint * 0.01
    target = _xy_to_xyz(tx, ty)
    if fixed_cct:
        sx, sy = cct_to_xy(fixed_cct)
        return adapt_matrix(_xy_to_xyz(sx, sy), target), {"fixed_cct": fixed_cct}
    est = estimate_neutral(rgb, exposure)
    info = {}
    if est is None:
        est = D65
    # limit how far we trust the estimate (mixed lighting, coloured walls)
    ex, ey = est[0] / est.sum(), est[1] / est.sum()
    info["est_xy"] = (round(float(ex), 4), round(float(ey), 4))
    # blend estimate toward D65 by (1 - strength): partial correction
    src = D65 * (1 - strength) + (est / est[1]) * strength
    # clamp shift magnitude
    def mired_of(xyz):
        x, y = xyz[0] / xyz.sum(), xyz[1] / xyz.sum()
        n = (x - 0.3320) / (0.1858 - y)
        cct = 449 * n**3 + 3525 * n**2 + 6823.3 * n + 5520.33
        return 1e6 / cct
    # Shift needed from the measured neutral to the target (mired). Cooling a warm
    # interior is allowed generously; warming a sky-lit scene only a little, so
    # exteriors keep blue skies and true-to-life shade.
    need = mired_of(target) - mired_of(src)
    lim = max_shift_mired if need < 0 else max_warm_mired
    if abs(need) > lim:
        tgt_m = mired_of(src) + np.sign(need) * lim
        cct = 1e6 / tgt_m
        tx2, ty2 = cct_to_xy(float(np.clip(cct, 2500, 12000)))
        target = _xy_to_xyz(tx2, ty2 + tint * 0.01)
    info["shift_mired"] = round(float(np.clip(need, -lim, lim)), 1)
    M = adapt_matrix(src / src[1], target / target[1])
    info["matrix"] = M.round(4).tolist()
    return M, info


# ---------------------------------------------------------------------------
# Tone
# ---------------------------------------------------------------------------

def tone_curve(v: np.ndarray, black: float, white: float, contrast: float,
               pivot: float = 0.42, toe: float = 0.04, shoulder: float = 0.10) -> np.ndarray:
    """Display-space S-curve. v in [0,1] (sRGB-encoded). black/white set the
    end points; contrast > 0 steepens around the pivot."""
    x = np.clip((v - black) / max(white - black, 1e-3), 0, None)
    # soft toe
    x = np.where(x < toe, toe * (x / toe) ** 1.6 if toe > 0 else x, x)
    # sigmoid contrast around pivot (normalised so 0->0, 1->1)
    if contrast != 0:
        k = 1 + contrast
        p = pivot
        lo = x <= p
        y = np.empty_like(x)
        y[lo] = p * (x[lo] / p) ** k
        xh = np.clip(x[~lo], p, None)
        y[~lo] = 1 - (1 - p) * ((1 - np.minimum(xh, 1)) / (1 - p)) ** k
        y[~lo] += np.clip(x[~lo] - 1, 0, None)  # keep >1 values monotone
        x = y
    # shoulder: soft roll-off into white
    if shoulder > 0:
        s0 = 1 - shoulder
        hi = x > s0
        t = (x[hi] - s0) / shoulder
        x[hi] = s0 + shoulder * (1 - np.exp(-t * 1.2)) / (1 - np.exp(-1.2))
    return np.clip(x, 0, 1)


def fast_blur(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur; large sigmas are computed on a downsampled copy (the result
    is smooth, so resampling loses nothing visible)."""
    if sigma < 12:
        return cv2.GaussianBlur(x, (0, 0), sigma)
    f = int(min(8, sigma // 6))
    small = cv2.resize(x, (x.shape[1] // f, x.shape[0] // f), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / f)
    return cv2.resize(small, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_LINEAR)


def clarity(Yd: np.ndarray, amount: float, sigma: float) -> np.ndarray:
    """Midtone local contrast on display luminance."""
    if amount <= 0:
        return Yd
    blur = fast_blur(Yd, sigma)
    detail = Yd - blur
    # weight midtones; fade out in bright areas (skies, views, white walls) so
    # clouds never turn crunchy and walls never get blotchy
    mid = np.clip(1 - np.abs(Yd - 0.45) / 0.45, 0, 1)
    hi_fade = 1 - np.clip((Yd - 0.70) / 0.22, 0, 1)
    return np.clip(Yd + amount * detail * (0.15 + 0.85 * mid) * hi_fade, 0, 1)


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------

def colour_grade(lab: np.ndarray, p: dict) -> np.ndarray:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    h = np.arctan2(b, a)  # radians
    # vibrance: lift low-chroma colours more than saturated ones
    vib = p.get("vibrance", 0.0)
    sat = p.get("saturation", 0.0)
    gain = 1 + sat + vib * np.exp(-C / 0.06)
    # keep near-black and near-white clean
    gain = 1 + (gain - 1) * np.clip(L / 0.25, 0, 1) * np.clip((1.02 - L) / 0.12, 0, 1)

    cmask = np.clip(C / 0.03, 0, 1)

    def hue_w(center_deg, width_deg):
        c = np.float32(np.deg2rad(center_deg))
        d = np.mod(h - c + np.float32(np.pi), np.float32(2 * np.pi)) - np.float32(np.pi)
        return ne.evaluate("exp(-(d / wd) ** 2) * cm", local_dict={"d": d, "wd": np.float32(np.deg2rad(width_deg)), "cm": cmask})

    # Sky blues (Oklab hue ~ 240-265 deg): a touch richer and slightly deeper
    wb_ = hue_w(250, 22)
    gain = gain * (1 + p.get("sky_sat", 0.0) * wb_)
    L = L - p.get("sky_deepen", 0.0) * wb_ * np.clip((L - 0.45) / 0.4, 0, 1)
    # Foliage: pull cyan-greens toward warm yellow-green
    wg = hue_w(140, 25)
    h = h + np.deg2rad(p.get("green_warm_deg", 0.0)) * wg
    gain = gain * (1 + p.get("green_sat", 0.0) * wg)
    # Timber/skin oranges (~55-75 deg): protect from over-saturation
    wo = hue_w(65, 25)
    gain = 1 + (gain - 1) * (1 - p.get("orange_protect", 0.0) * wo)

    C2 = C * gain
    a2, b2 = C2 * np.cos(h), C2 * np.sin(h)
    # global warmth: small shift of whites/mids toward yellow, shadows kept neutral
    warm = p.get("warmth_b", 0.0)
    a2 = a2 + p.get("tint_a", 0.0) * np.clip(L, 0, 1)
    b2 = b2 + warm * np.clip((L - 0.15) / 0.6, 0, 1) * (1 - 0.9 * wb_)
    return np.stack([L, a2, b2], axis=-1)


def gamut_map_srgb(rgb2020: np.ndarray) -> np.ndarray:
    """Rec.2020 linear -> sRGB linear with soft chroma compression toward the
    pixel's luminance instead of hard per-channel clipping."""
    s = _mat(rgb2020, REC2020_TO_SRGB)
    Y = luminance(rgb2020)[..., None]
    lo = s.min(axis=-1, keepdims=True)
    # amount of desaturation needed to bring negatives to 0
    need = np.where(lo < 0, Y / np.maximum(Y - lo, 1e-6), 1.0)
    need = np.clip(need, 0, 1)
    s = Y + (s - Y) * need
    return np.clip(s, 0, None)


def sharpen(v: np.ndarray, radius: float, amount: float, threshold: float = 0.01) -> np.ndarray:
    """Luminance-only unsharp mask on an sRGB-encoded HxWx3 image with an edge
    threshold so flat surfaces (walls, sky) stay clean."""
    Y = v @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    blur = cv2.GaussianBlur(Y, (0, 0), radius)
    d = Y - blur
    m = np.clip((np.abs(d) - threshold) / threshold, 0, 1)
    return np.clip(v + (amount * d * m)[..., None], 0, 1)
