"""Bracket grouping, alignment and linear-light HDR merge with ghost suppression."""
from __future__ import annotations

import datetime as dt
from collections import Counter

import cv2
import numpy as np

from .raw import Frame

LUMA = np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)  # Rec.2020 luminance


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ LUMA


def _ts(m: dict) -> float:
    s = str(m.get("DateTimeOriginal", "1970:01:01 00:00:00"))[:19]
    t = dt.datetime.strptime(s, "%Y:%m:%d %H:%M:%S").timestamp()
    sub = str(m.get("SubSecTimeOriginal") or "0")
    return t + float("0." + sub.strip()) if sub.strip().isdigit() else t


def group_brackets(metas: list[dict], max_gap: float = 1.5) -> list[list[dict]]:
    """Group frames into exposure brackets by capture time and shared settings."""
    ms = sorted(metas, key=_ts)
    clusters: list[list[dict]] = []
    for m in ms:
        if clusters:
            prev = clusters[-1][-1]
            same = (m.get("FNumber") == prev.get("FNumber") and m.get("ISO") == prev.get("ISO")
                    and abs(float(m.get("FocalLength") or 0) - float(prev.get("FocalLength") or 0)) < 1.5
                    and m.get("Model") == prev.get("Model"))
            if same and _ts(m) - _ts(prev) <= max_gap:
                clusters[-1].append(m)
                continue
        clusters.append([m])
    # Split clusters longer than the typical bracket size (back-to-back brackets).
    sizes = Counter(len(c) for c in clusters if len(c) > 1)
    k = sizes.most_common(1)[0][0] if sizes else 1
    out = []
    for c in clusters:
        if len(c) > k and len(c) % k == 0:
            out.extend(c[i:i + k] for i in range(0, len(c), k))
        else:
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def _align_img(f: Frame, ref_exposure: float, size: int) -> tuple[np.ndarray, np.ndarray, float]:
    y = luminance(f.rgb) * (ref_exposure / f.exposure)
    s = size / max(y.shape)
    small = cv2.resize(y, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    clip = cv2.resize(f.clip, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    lo = 0.003 * (ref_exposure / f.exposure)  # noise floor of this frame in ref units
    valid = ((clip < 0.05) & (small > lo)).astype(np.uint8)
    return np.log2(np.maximum(small, 1e-6)).astype(np.float32), valid, s


def align(frames: list[Frame], ref_idx: int, motion: str = "auto", size: int = 2000) -> list[Frame]:
    """Align all frames to frames[ref_idx] (in place). Returns frames."""
    ref = frames[ref_idx]
    ry, rv, s = _align_img(ref, ref.exposure, size)
    mode = {"translation": cv2.MOTION_TRANSLATION, "euclidean": cv2.MOTION_EUCLIDEAN,
            "homography": cv2.MOTION_HOMOGRAPHY}
    for i, f in enumerate(frames):
        if i == ref_idx:
            continue
        fy, fv, _ = _align_img(f, ref.exposure, size)
        mask = (rv & fv)
        if mask.mean() < 0.05:
            continue
        # Fill invalid pixels so they do not drive the correlation
        a = np.where(mask > 0, ry, np.median(ry[mask > 0])).astype(np.float32)
        b = np.where(mask > 0, fy, np.median(fy[mask > 0])).astype(np.float32)
        a = cv2.GaussianBlur(a, (0, 0), 1.0)
        b = cv2.GaussianBlur(b, (0, 0), 1.0)
        mtype = mode["homography" if motion == "auto" and f.meta.get("Make", "").upper().startswith("DJI")
                     else ("euclidean" if motion == "auto" else motion)]
        warp = np.eye(3, dtype=np.float32) if mtype == cv2.MOTION_HOMOGRAPHY else np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(a, b, warp, mtype,
                                           (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6),
                                           mask, 5)
        except cv2.error:
            continue
        # scale to full resolution
        S = np.diag([1 / s, 1 / s, 1]).astype(np.float64)
        Wm = warp.astype(np.float64)
        if Wm.shape[0] == 2:
            Wm = np.vstack([Wm, [0, 0, 1]])
        Wf = S @ Wm @ np.linalg.inv(S)
        # displacement magnitude at the corners
        H, W = f.rgb.shape[:2]
        corners = np.array([[0, 0, 1], [W, 0, 1], [0, H, 1], [W, H, 1]], dtype=np.float64).T
        moved = Wf @ corners
        moved = moved[:2] / moved[2]
        disp = np.abs(moved - corners[:2]).max()
        f.meta["align_disp_px"] = float(disp)
        if disp < 0.35:
            continue
        flags = cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP
        dt = f.rgb.dtype
        f.rgb = cv2.warpPerspective(f.rgb.astype(np.float32), Wf, (W, H), flags=flags,
                                    borderMode=cv2.BORDER_REFLECT).astype(dt)
        f.clip = cv2.warpPerspective(f.clip, Wf, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_REFLECT)
        if f.hard is not None:
            f.hard = cv2.warpPerspective(f.hard, Wf, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                         borderMode=cv2.BORDER_REFLECT)
    return frames


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def merge(frames: list[Frame], deghost: bool = True) -> tuple[np.ndarray, dict]:
    """Merge aligned frames into a linear HDR radiance map (units of the
    reference=middle exposure). Returns (hdr, info)."""
    order = np.argsort([f.exposure for f in frames])  # short -> long
    frames = [frames[i] for i in order]
    ref_i = len(frames) // 2
    ref = frames[ref_i]
    info = {"exposures": [f.exposure for f in frames], "ratios": []}

    # Refine exposure ratios from the data (shutter speeds are nominal).
    Yref = luminance(ref.rgb)
    scales = []
    for f in frames:
        nominal = ref.exposure / f.exposure
        if f is ref:
            scales.append(1.0)
            continue
        Yf = luminance(f.rgb) * nominal
        step = max(1, int(np.sqrt(Yref.size / 2e6)))
        a, b = Yref[::step, ::step], Yf[::step, ::step]
        ca, cb = ref.clip[::step, ::step], f.clip[::step, ::step]
        ok = (ca < 0.01) & (cb < 0.01) & (a > 0.02) & (a < 0.6) & (b / nominal > 0.02) & (b / nominal < 0.6)
        if ok.sum() > 2000:
            corr = float(np.median(a[ok] / np.maximum(b[ok], 1e-9)))
            corr = float(np.clip(corr, 0.8, 1.25))
        else:
            corr = 1.0
        scales.append(nominal * corr)
        info["ratios"].append(round(corr, 4))

    H, W = ref.rgb.shape[:2]
    acc = np.zeros((H, W, 3), np.float32)
    tmp = np.empty((H, W, 3), np.float32)
    wsum = np.zeros((H, W), np.float32)

    # Ghost reference: reference frame, falling back to shorter frames where it clips.
    if deghost:
        gref = luminance(ref.rgb) * scales[ref_i]
        gvalid = 1 - ref.clip
        for j in range(ref_i - 1, -1, -1):
            yj = luminance(frames[j].rgb) * scales[j]
            fill = (gvalid < 0.5)
            gref = np.where(fill, yj, gref)
            gvalid = np.where(fill, 1 - frames[j].clip, gvalid)
        gref_log = cv2.GaussianBlur(np.log2(np.maximum(gref, 1e-5)), (0, 0), 3)

    for k, f in enumerate(frames):
        y = luminance(f.rgb)
        ys = cv2.GaussianBlur(y, (0, 0), 2.0)
        # usable range: f.clip already rolls off smoothly toward sensor clip;
        # also drop this frame's deep noise floor (on a smoothed signal)
        w = (1 - f.clip) * _smoothstep(0.0005, 0.004, ys)
        w *= f.exposure / frames[-1].exposure  # SNR: favour longer exposures
        if k == 0:
            w = np.maximum(w, 1e-4)  # shortest frame always contributes (fully clipped areas)
        if deghost and k != ref_i:
            yl = cv2.GaussianBlur(np.log2(np.maximum(y * scales[k], 1e-5)), (0, 0), 3)
            # noise-aware tolerance: generous in this frame's shadows
            tol = 0.35 + 0.6 * (1 - _smoothstep(0.003, 0.03, ys))
            dev = np.abs(yl - gref_log)
            ghost = _smoothstep(tol, tol + 0.5, dev)
            ghost = cv2.GaussianBlur(ghost, (0, 0), 4)
            w *= (1 - ghost)
            if k == 0:
                w = np.maximum(w, 1e-4)
        np.multiply(f.rgb, (w * np.float32(scales[k]))[:, :, None], out=tmp)
        acc += tmp
        wsum += w
    del tmp
    acc /= np.maximum(wsum, 1e-8)[:, :, None]
    hdr = acc
    # Highlights clipped even in the shortest frame (sun, glare on water): the
    # surviving channel ratios are meaningless and turn magenta after white
    # balance. Render them as neutral white at the brightest channel's level.
    c = frames[0].clip
    if frames[0].hard is not None:
        # every pixel that clipped even in the darkest frame goes fully neutral
        hard = cv2.GaussianBlur(np.clip(frames[0].hard * 1.5, 0, 1), (0, 0), 1.0)
        c = np.maximum(c, hard)
    if c.max() > 0:
        mx = hdr.max(axis=2, keepdims=True)
        cc = c[:, :, None]
        hdr = hdr * (1 - cc) + mx * cc
    info["scales"] = [round(s, 5) for s in scales]
    return hdr.astype(np.float32), info
