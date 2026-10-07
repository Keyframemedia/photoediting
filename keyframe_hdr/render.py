"""HDR radiance -> finished, display-ready sRGB image."""
from __future__ import annotations

import numpy as np
import cv2

from . import grade, tonemap
from .merge import luminance


def render(hdr: np.ndarray, p: dict, return_info: bool = False):
    info = {}
    Y = luminance(hdr)
    # 1. exposure: anchor the room's whites, keep the median in a sane range
    if p.get("exposure_mode", "anchored") == "anchored":
        k, ei = tonemap.anchored_exposure(Y, white_target=p.get("white_anchor", 0.78),
                                          white_pct=p.get("white_anchor_pct", 99.0),
                                          window_stops=p.get("window_stops", 2.5),
                                          median_range=tuple(p.get("median_range", (0.07, 0.22))))
        info.update(ei)
    else:
        k = tonemap.auto_exposure(Y, key=p["key"], pct=p.get("key_pct", 50),
                                  exclude_top=p.get("key_exclude_top", 0.08))
    k *= 2.0 ** p.get("exposure_bias", 0.0)
    info["exposure"] = float(k)
    rgb = hdr * k

    # 2. white balance toward the house white
    if p.get("wb_strength", 0) > 0 or p.get("wb_target_cct"):
        rgb, wbi = grade.white_balance(rgb, 1.0, p["wb_target_cct"], p.get("wb_strength", 0.7),
                                       tint=p.get("wb_tint", 0.0),
                                       max_shift_mired=p.get("wb_max_mired", 45),
                                       max_warm_mired=p.get("wb_max_warm_mired", 12),
                                       fixed_cct=p.get("wb_fixed_cct"))
        info["wb"] = wbi
    Y = luminance(rgb)

    # 3. local tone mapping (exposure fusion of virtual exposures)
    Yd = tonemap.fuse_luminance(Y, evs=tuple(p["fusion_evs"]), sigma=p["fusion_sigma"],
                                center=p.get("fusion_center", 0.5),
                                contrast_weight=p.get("fusion_contrast_weight", 0.0),
                                mode=p.get("fusion_mode", "hat"), ev_prior=p.get("fusion_ev_prior", 0.6),
                                hat=tuple(p.get("fusion_hat", (0.03, 0.14, 0.86, 0.985))))
    # blend some global (non-local) rendering back for depth: pure fusion is flat
    Yg = tonemap.srgb_encode(np.clip(Y / (1 + Y / p.get("global_white", 6.0)), 0, 1))
    Yd = (1 - p.get("global_mix", 0.0)) * Yd + p.get("global_mix", 0.0) * Yg

    # 4. levels: black point from the image (Blacks slider to the clipping point);
    #    whites only nudged up if the brightest areas fall short, never squashed.
    if p.get("levels", True):
        st = (max(1, Yd.shape[0] // 700), max(1, Yd.shape[1] // 700))
        smp = Yd[::st[0], ::st[1]]
        lo = min(float(np.percentile(smp, p.get("black_pct", 0.3))), 0.06)
        hi = float(np.percentile(smp, p.get("white_pct", 99.5)))
        tgt_hi = p.get("white_target", 0.96)
        gain = float(np.clip(tgt_hi / max(hi - lo, 1e-3), 1.0, p.get("max_white_stretch", 1.12)))
        Yd = np.clip((Yd - lo) * gain, 0, 1)
        info["levels"] = (round(lo, 3), round(hi, 3), round(gain, 3))
    Yd = grade.tone_curve(Yd, p["black"], p["white"], p["contrast"], pivot=p.get("pivot", 0.42),
                          toe=p.get("toe", 0.04), shoulder=p.get("shoulder", 0.1))
    sig = p.get("clarity_sigma_frac", 0.012) * max(Yd.shape)
    Yd = grade.clarity(Yd, p.get("clarity", 0.0), sig)
    if p.get("micro_contrast", 0) > 0:
        Yd = grade.clarity(Yd, p["micro_contrast"], p.get("micro_sigma_frac", 0.0025) * max(Yd.shape))

    Yd_lin = tonemap.srgb_decode(Yd).astype(np.float32)
    rgb = tonemap.apply_luminance(rgb, Y, Yd_lin, desat_highlights=p.get("desat_highlights", 0.6))

    # 5. colour in Oklab
    lab = grade.rec2020_to_oklab(rgb)
    lab = grade.colour_grade(lab, p)
    rgb = grade.oklab_to_rec2020(lab)

    # 6. to sRGB
    s = grade.gamut_map_srgb(rgb.astype(np.float32))
    v = tonemap.srgb_encode(np.clip(s, 0, 1)).astype(np.float32)
    if return_info:
        return v, info
    return v


def output_sharpen(v: np.ndarray, p: dict, long_edge: int | None = None) -> np.ndarray:
    """Resize (optional) then apply output sharpening tuned for that size."""
    if long_edge and max(v.shape[:2]) > long_edge:
        s = long_edge / max(v.shape[:2])
        v = cv2.resize(v, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    r = p.get("sharpen_radius", 0.8)
    a = p.get("sharpen_amount", 0.6)
    if long_edge and long_edge <= 3000:
        r, a = p.get("web_sharpen_radius", 0.6), p.get("web_sharpen_amount", 0.5)
    return grade.sharpen(v, r, a, p.get("sharpen_threshold", 0.008))
