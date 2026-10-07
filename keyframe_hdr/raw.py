"""RAW decoding to linear, white-balanced, lens-corrected scene-referred RGB.

Output frames are float32 linear Rec.2020, scaled so that a value of 1.0 is the
sensor clip point of the brightest channel at this frame's exposure, plus a soft
saturation mask (1 = clipped, unusable) in the same geometry.
"""
from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass

import cv2
import numpy as np
import rawpy

from . import lens

# linear sRGB (D65) -> linear Rec.2020 (D65)
SRGB_TO_REC2020 = np.array([
    [0.6274040, 0.3292820, 0.0433136],
    [0.0690970, 0.9195400, 0.0113612],
    [0.0163916, 0.0880132, 0.8955950],
], dtype=np.float32)


@dataclass
class Frame:
    path: str
    rgb: np.ndarray        # HxWx3 float32 linear Rec.2020
    clip: np.ndarray       # HxW float32, 1 where any channel was near sensor clip
    exposure: float        # relative exposure (shutter * ISO / N^2)
    meta: dict
    hard: np.ndarray | None = None  # HxW float32, 1 where a channel actually hit clip (unblurred)


def exif(paths: list[str]) -> list[dict]:
    tags = ["-FileName", "-Directory", "-DateTimeOriginal", "-SubSecTimeOriginal",
            "-ExposureTime", "-FNumber", "-ISO", "-ExposureCompensation",
            "-FocalLength", "-FocusDistance", "-LensModel", "-Model", "-Make",
            "-Orientation", "-VignetteCoefficient1", "-VignetteCoefficient2",
            "-VignetteCoefficient3", "-NEFCompression", "-ImageWidth", "-ImageHeight"]
    out = subprocess.run(["exiftool", "-j", "-n", *tags, *paths],
                         capture_output=True, text=True, check=False).stdout
    return json.loads(out) if out.strip() else []


def relative_exposure(m: dict) -> float:
    t = float(m.get("ExposureTime") or 1.0)
    iso = float(m.get("ISO") or 100.0)
    n = float(m.get("FNumber") or 8.0)
    return t * iso / (n * n)


def decode(path: str, meta: dict, nef_meta: dict | None = None, half: bool = False,
           wb: list | None = None, nikon_vignette_strength: float = 0.75) -> Frame:
    """Decode one DNG to linear Rec.2020 with lens corrections applied."""
    with rawpy.imread(path) as r:
        cam_wb = list(r.camera_whitebalance[:3]) if wb is None else list(wb[:3])
        if cam_wb[0] <= 0:
            cam_wb = list(r.daylight_whitebalance[:3])
        # Normalise multipliers by their max so nothing clips when WB is applied
        # (HighlightMode.Ignore keeps the full range); data is then scaled back up.
        mmax = max(cam_wb)
        rgb = r.postprocess(
            output_color=rawpy.ColorSpace.raw, gamma=(1, 1), no_auto_bright=True,
            output_bps=16, user_wb=cam_wb + [cam_wb[1]],
            highlight_mode=rawpy.HighlightMode.Ignore, adjust_maximum_thr=0.0,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.DHT, half_size=half,
            user_flip=0, fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Off,
        )
        cm = r.color_matrix[:, :3].astype(np.float32)
        sizes = r.sizes
    rgb = rgb.astype(np.float32) * (1.0 / 65535.0)
    # Per-channel clip level after WB normalisation by the max multiplier.
    clip_lvl = np.array([w / mmax for w in cam_wb], dtype=np.float32)
    # Saturation relative to each channel's own clip point. Made spatially smooth
    # (dilate + blur) so merge weights never flip pixel-to-pixel on noise, and
    # conservative around clipped areas where demosaicing has mixed in bad values.
    sat = np.max(rgb / clip_lvl[None, None, :], axis=2).astype(np.float32)
    # Pixels where a channel truly hit the sensor's clip point: their colour is
    # unrecoverable (glints, LEDs, the sun). Kept sharp so the merge can render
    # them white when even the darkest frame clipped, without touching real colour.
    hard = cv2.dilate((sat >= 0.97).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    sat = cv2.dilate(sat, np.ones((5, 5), np.uint8))
    sat = cv2.GaussianBlur(sat, (0, 0), 2.0 if not half else 1.0)
    t = np.clip((sat - 0.80) / 0.16, 0, 1)
    clip = (t * t * (3 - 2 * t)).astype(np.float32)
    # bring brightest-channel clip point to 1.0
    rgb *= 1.0 / max(clip_lvl)

    H, W = rgb.shape[:2]
    scale = 0.5 if half else 1.0

    ops = lens.read_opcode_list3(path)
    if ops.gainmaps:
        lens.apply_gainmaps(rgb, ops.gainmaps, scale)
    else:
        g = None
        lm = (nef_meta or meta).get("LensModel") or ""
        lcp = lens.find_lcp(lm)
        if lcp:
            g = lens.lcp_vignette_gain(lcp, float(meta.get("FocalLength") or 24),
                                       float(meta.get("FNumber") or 8),
                                       float(meta.get("FocusDistance") or 10), H, W)
        elif nef_meta and nef_meta.get("VignetteCoefficient1") is not None and "NIKKOR" in lm.upper():
            g = lens.nikon_vignette_gain(
                (float(nef_meta["VignetteCoefficient1"]), float(nef_meta["VignetteCoefficient2"]),
                 float(nef_meta["VignetteCoefficient3"])), H, W)
            if g is not None:
                g = np.power(g, nikon_vignette_strength)
        if g is not None:
            rgb *= g[:, :, None]
    if ops.warp is not None:
        rgb, masks = lens.apply_warp(rgb, ops.warp, np.dstack([clip, hard]))
        clip, hard = np.ascontiguousarray(masks[..., 0]), np.ascontiguousarray(masks[..., 1])

    # default crop (LibRaw leaves the DNG DefaultCrop margins in place)
    cl, ct = sizes.crop_left_margin, sizes.crop_top_margin
    cw, ch = sizes.crop_width, sizes.crop_height
    if cw and ch:
        if half:
            cl, ct, cw, ch = cl // 2, ct // 2, cw // 2, ch // 2
        rgb = rgb[ct:ct + ch, cl:cl + cw]
        clip = clip[ct:ct + ch, cl:cl + cw]
        hard = hard[ct:ct + ch, cl:cl + cw]

    # camera RGB -> linear sRGB -> linear Rec.2020
    M = SRGB_TO_REC2020 @ cm
    rgb = (rgb.reshape(-1, 3) @ M.T).reshape(rgb.shape).astype(np.float32)
    np.maximum(rgb, 0, out=rgb)

    orient = int(meta.get("Orientation") or 1)
    turn = {6: lambda a: np.rot90(a, -1), 8: lambda a: np.rot90(a, 1), 3: lambda a: a[::-1, ::-1]}.get(orient)
    if turn:
        rgb, clip, hard = (np.ascontiguousarray(turn(a)) for a in (rgb, clip, hard))

    return Frame(path, rgb, clip, relative_exposure(meta), dict(meta, wb=cam_wb), hard)
