"""HDR radiance -> finished, display-ready sRGB image."""
from __future__ import annotations

import numpy as np
import cv2

from . import grade, tonemap
from .merge import luminance


STRIP = 512  # rows per strip for the pixel-wise colour stages (bounds peak memory)


def render(hdr: np.ndarray, p: dict, return_info: bool = False, inplace: bool = False,
           sky: np.ndarray | None = None):
    """inplace=True reuses (and destroys) `hdr`'s buffer - the pipeline uses this
    to keep peak memory down at 45 MP. `sky` (optional, full-size alpha) lets the
    grade treat the sky on its own (twilight purple look)."""
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
    if inplace:
        rgb = hdr
        rgb *= np.float32(k)
    else:
        rgb = hdr * np.float32(k)
    del Y

    # 2. white balance toward the house white (matrix measured globally, applied in strips)
    if p.get("wb_strength", 0) > 0 or p.get("wb_target_cct"):
        M, wbi = grade.wb_matrix(rgb, 1.0, p["wb_target_cct"], p.get("wb_strength", 0.7),
                                 tint=p.get("wb_tint", 0.0),
                                 max_shift_mired=p.get("wb_max_mired", 45),
                                 max_warm_mired=p.get("wb_max_warm_mired", 12),
                                 fixed_cct=p.get("wb_fixed_cct"))
        Mt = M.T.astype(np.float32)
        for r0 in range(0, rgb.shape[0], STRIP):
            sl = rgb[r0:r0 + STRIP]
            np.maximum(sl @ Mt, 0, out=sl)
        info["wb"] = wbi
    Y = luminance(rgb)

    # 3. local tone mapping (exposure fusion of virtual exposures)
    Yd = tonemap.fuse_luminance(Y, evs=tuple(p["fusion_evs"]), sigma=p["fusion_sigma"],
                                center=p.get("fusion_center", 0.5),
                                contrast_weight=p.get("fusion_contrast_weight", 0.0),
                                mode=p.get("fusion_mode", "hat"), ev_prior=p.get("fusion_ev_prior", 0.6),
                                hat=tuple(p.get("fusion_hat", (0.03, 0.14, 0.86, 0.985))))
    # Bright regions (sky, views) take a single *global* exposure instead of the
    # locally adapted fusion: keeps natural cloud-to-blue contrast (no grey,
    # crunchy "HDR skies") while rooms keep the local balancing. Weighting is a
    # per-pixel function of luminance only, so it cannot create halos.
    if p.get("sky_global", 0) > 0:
        lY = np.log2(np.maximum(Y, 1e-7))
        lk = np.log2(tonemap_key(Y, p))
        bright = tonemap._smoothstep(lk + p.get("sky_global_lo", 1.2), lk + p.get("sky_global_hi", 2.6), lY)
        sel = bright[:: max(1, Y.shape[0] // 500), :: max(1, Y.shape[1] // 500)] > 0.5
        ysm = Y[:: max(1, Y.shape[0] // 500), :: max(1, Y.shape[1] // 500)]
        if sel.sum() > 200:
            # anchor the sky's median (not its brightest point - that is the sun)
            ref = float(np.percentile(ysm[sel], p.get("sky_global_pct", 50.0)))
            x = Y * (p.get("sky_global_white", 0.60) / max(ref, 1e-7))
            n = p.get("sky_global_shoulder", 4.0)
            x = x / np.power(1 + np.power(x, n), 1 / n)  # soft shoulder into white
            Yg = tonemap.srgb_encode(x)
            w = bright * p["sky_global"]
            Yd = Yd * (1 - w) + Yg * w
            info["sky_global_ref"] = round(ref, 4)

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
    del Yd
    st = (max(1, Y.shape[0] // 500), max(1, Y.shape[1] // 500))
    ratio_ref = float(np.percentile(Yd_lin[::st[0], ::st[1]] / np.maximum(Y[::st[0], ::st[1]], 1e-7), 60))

    # 5-6. per-pixel stages in strips, written back into rgb's buffer:
    #      luminance -> colour (Oklab) -> gamut map -> sRGB encode
    desat = p.get("desat_highlights", 0.6)
    house = None
    if p.get("lut"):  # the house colour rendering, learned from delivered images (lut.py)
        from . import lut as lutmod
        house = lutmod.load(p["lut"])
        info["lut"] = p["lut"] if house is not None else f"missing:{p['lut']}"
    for r0 in range(0, rgb.shape[0], STRIP):
        sl = slice(r0, r0 + STRIP)
        c = tonemap.apply_luminance(rgb[sl], Y[sl], Yd_lin[sl], desat_highlights=desat, ratio_ref=ratio_ref)
        lab = grade.colour_grade(grade.rec2020_to_oklab(c), p, sky=None if sky is None else sky[sl])
        c = grade.gamut_map_srgb(grade.oklab_to_rec2020(lab).astype(np.float32))
        rgb[sl] = tonemap.srgb_encode(np.clip(c, 0, 1))
        if house is not None:
            lutmod.apply(rgb[sl], house, p.get("lut_strength", 1.0))
    v = rgb
    if p.get("despeckle", False):
        v, info["specks"] = grade.suppress_specular_specks(v)
    if return_info:
        return v, info
    return v


def tonemap_key(Y: np.ndarray, p: dict) -> float:
    """Median luminance of the non-window part of the (already exposed) image."""
    s = Y[:: max(1, Y.shape[0] // 400), :: max(1, Y.shape[1] // 400)].ravel()
    s = s[s > 0]
    hi = np.percentile(s, 100 * (1 - p.get("key_exclude_top", 0.1)))
    return float(np.median(s[s < hi]))


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
