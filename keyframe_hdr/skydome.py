"""Sky replacement from 360-degree sky domes.

Every exterior of a property is given the *same* physical sky, seen from that
shot's own viewing direction. Turn the camera from south to west and the clouds
change exactly as they would have on the day, so a set never shows the same
cloud pasted into every frame.

Domes are CC0 "pure sky" HDRIs from Poly Haven (polyhaven.com, no attribution
required, commercial use allowed): equirectangular, scene-referred linear light,
so they composite into the linear HDR before the house grade, like a real sky.

Viewing direction per shot:
  * drone (DJI): the gimbal's compass yaw, relative to the first shot of the job;
  * cameras without a compass (Canon, Nikon): a golden-angle sequence over the
    shot index, so consecutive frames look ~85 degrees apart, within a range
    that keeps the sun behind the photographer (front-lit facades, deepest blue).
Pitch, roll, focal length and principal point come from the upright step, so
the horizon of the dome lands on the real horizon.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess

import cv2
import numpy as np

CACHE = os.environ.get("KF_SKY_CACHE", os.path.expanduser("~/.cache/keyframe_skies"))
RES = os.environ.get("KF_SKY_RES", "16k")

# option -> candidate domes; a property gets one of them (stable per job seed)
LIBRARY = {
    "clouds": ["kloofendal_48d_partly_cloudy_puresky", "kloofendal_38d_partly_cloudy_puresky"],
    "clear": ["syferfontein_18d_clear_puresky", "kloofendal_43d_clear_puresky"],
    "twilight_clear": ["rosendal_park_sunset_puresky"],
    "twilight_purple": ["qwantani_dusk_2_puresky"],
    "twilight_clouds": ["belfast_sunset_puresky"],
}
# viewing direction relative to the dome's sun: (centre, half range) in degrees.
# Daylight looks away from the sun; twilight keeps the afterglow off to one side.
AIM = {"clouds": (180.0, 70.0), "clear": (180.0, 70.0),
       "twilight_clear": (140.0, 60.0), "twilight_purple": (140.0, 60.0),
       "twilight_clouds": (140.0, 60.0)}

BAND_TOP, BAND_BOTTOM = 90.0, -10.0  # elevation range kept from each dome (deg)
GOLDEN = 0.6180339887

# linear sRGB / Rec.709 (the domes) -> linear Rec.2020 (the pipeline)
_709_TO_2020 = np.array([[0.6274040, 0.3292820, 0.0433136],
                         [0.0690970, 0.9195400, 0.0113612],
                         [0.0163916, 0.0880132, 0.8955950]], dtype=np.float32)


def choose(option: str, seed: str | int = 0) -> str:
    names = LIBRARY[option]
    h = int(hashlib.sha1(str(seed).encode()).hexdigest(), 16)
    return names[h % len(names)]


def fetch(name: str, res: str = RES) -> str:
    """Download a dome (once) and return the local .hdr path."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{name}_{res}.hdr")
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    info = json.loads(subprocess.run(["curl", "-sS", "-m", "60", f"https://api.polyhaven.com/files/{name}"],
                                     capture_output=True, text=True, check=True).stdout)
    url = info["hdri"][res]["hdr"]["url"]
    tmp = path + ".part"
    subprocess.run(["curl", "-sS", "-L", "--retry", "4", "-m", "900", "-o", tmp, url], check=True)
    os.replace(tmp, path)
    return path


def load(name: str, res: str = RES) -> tuple[np.ndarray, dict]:
    """The dome's upper band (elevation 90 .. -10 deg) as float16 linear Rec.2020,
    memory-mapped from a cache file, plus its sun position."""
    npy = os.path.join(CACHE, f"{name}_{res}_band.npy")
    meta_p = npy[:-4] + ".json"
    if not (os.path.exists(npy) and os.path.exists(meta_p)):
        img = cv2.imread(fetch(name, res), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"could not read sky dome {name}")
        Hd, Wd = img.shape[:2]
        v1 = int(round((90.0 - BAND_BOTTOM) / 180.0 * Hd))
        band = np.empty((v1, Wd, 3), np.float16)
        for r0 in range(0, v1, 512):
            blk = img[r0:min(v1, r0 + 512), :, ::-1].astype(np.float32)
            band[r0:r0 + blk.shape[0]] = np.clip(blk @ _709_TO_2020.T, 0, 60000)  # float16 range
        del img
        # sun: brightest spot of a blurred low-res copy
        small = cv2.resize(band.astype(np.float32), (1024, int(1024 * v1 / Wd)), interpolation=cv2.INTER_AREA)
        Ys = cv2.GaussianBlur(small @ np.array([0.2627, 0.678, 0.0593], np.float32), (0, 0), 3)
        sy, sx = np.unravel_index(int(np.argmax(Ys)), Ys.shape)
        meta = {"name": name, "res": res, "width": Wd, "height": Hd, "rows": v1,
                "sun_az": (sx + 0.5) / small.shape[1] * 360.0 - 180.0,
                "sun_el": 90.0 - (sy + 0.5) / small.shape[0] * (90.0 - BAND_BOTTOM)}
        tmp = npy + ".part.npy"
        np.save(tmp, band)
        os.replace(tmp, npy)
        json.dump(meta, open(meta_p, "w"))
        del band
    return np.load(npy, mmap_mode="r"), json.load(open(meta_p))


def view_yaw(option: str, meta: dict, shot_meta: dict, shot_index: int, seed: str | int,
             ref_yaw: float | None = None) -> float:
    """Dome azimuth (deg) the camera looks along for this shot."""
    centre, half = AIM.get(option, (180.0, 70.0))
    s = (int(hashlib.sha1(str(seed).encode()).hexdigest(), 16) % 10_000) / 10_000.0
    base = meta["sun_az"] + centre + half * (2 * s - 1) * 0.5
    yaw = shot_meta.get("GimbalYawDegree")
    if yaw is None:
        yaw = shot_meta.get("FlightYawDegree")
    if yaw is not None and ref_yaw is not None:
        # a compass: rotate the dome exactly as the camera turned
        return _wrap(base + float(yaw) - float(ref_yaw))
    # no compass: golden-angle spread over the shot sequence
    t = math.fmod(shot_index * GOLDEN + s, 1.0)
    return _wrap(meta["sun_az"] + centre + half * (2 * t - 1))


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def camera_rotation(pitch_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """Camera -> level-world rotation (x right, y down, z forward) for a camera
    pitched up by `pitch_deg` and rolled clockwise by `roll_deg`."""
    p, r = math.radians(pitch_deg), math.radians(roll_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return Rx @ Rz


def render_view(band: np.ndarray, meta: dict, W: int, H: int, f_px: float, pp: tuple[float, float],
                R: np.ndarray, yaw_deg: float, rows: tuple[int, int] | None = None,
                strip: int = 768) -> np.ndarray:
    """Perspective view of the dome for an image of size W x H (only rows r0..r1),
    linear Rec.2020 float32, shape (r1 - r0, W, 3)."""
    r0, r1 = rows or (0, H)
    Wd, Hd, nb = meta["width"], meta["height"], band.shape[0]
    out = np.empty((r1 - r0, W, 3), np.float32)
    R = np.asarray(R, np.float64)
    xs = ((np.arange(W, dtype=np.float64) - pp[0]) / f_px)[None, :]
    for y0 in range(r0, r1, strip):
        y1 = min(r1, y0 + strip)
        ys = ((np.arange(y0, y1, dtype=np.float64) - pp[1]) / f_px)[:, None]
        dx = R[0, 0] * xs + R[0, 1] * ys + R[0, 2]
        dy = R[1, 0] * xs + R[1, 1] * ys + R[1, 2]
        dz = R[2, 0] * xs + R[2, 1] * ys + R[2, 2]
        el = np.degrees(np.arctan2(-dy, np.hypot(dx, dz)))
        az = np.degrees(np.arctan2(dx, dz)) + yaw_deg
        el = np.clip(el, BAND_BOTTOM + 1.0, 89.9)  # below the horizon: repeat the horizon haze
        u = np.mod((az + 180.0) / 360.0 * Wd, Wd)
        v = (90.0 - el) / 180.0 * Hd
        # cut the needed window out of the (memory-mapped) dome, handling the wrap
        vmin, vmax = int(max(0, np.floor(v.min()) - 3)), int(min(nb, np.ceil(v.max()) + 4))
        uc = np.degrees(np.arctan2(np.sin(np.radians(u / Wd * 360)).mean(),
                                   np.cos(np.radians(u / Wd * 360)).mean())) / 360 * Wd
        du = np.mod(u - uc + Wd / 2, Wd) - Wd / 2  # offset from the window centre
        umin, umax = int(np.floor(du.min()) - 3), int(np.ceil(du.max()) + 4)
        cols = np.mod(np.arange(umin, umax) + int(round(uc)), Wd)
        src = np.ascontiguousarray(band[vmin:vmax][:, cols].astype(np.float32))
        mx = (du - umin + (uc - round(uc))).astype(np.float32)
        my = (v - vmin).astype(np.float32)
        out[y0 - r0:y1 - r0] = cv2.remap(src, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return np.maximum(out, 0)


def geometry_for(ginfo: dict, ref_meta: dict, W: int, H: int) -> tuple[float, tuple[float, float], np.ndarray]:
    """Focal length (px), principal point and camera rotation for the image as it
    stands after the upright step."""
    cam = ginfo.get("camera") if ginfo else None
    if cam:
        return float(cam["f"]), tuple(cam["pp"]), np.asarray(cam["R"], np.float64)
    f35 = float(ref_meta.get("FocalLengthIn35mmFormat") or 0)
    if f35 > 0:
        f_px = f35 * math.hypot(W, H) / 43.27
    else:
        f_px = float(ref_meta.get("FocalLength") or 24) / 36.0 * max(W, H)
    pitch = ref_meta.get("GimbalPitchDegree")
    roll = ref_meta.get("GimbalRollDegree") or 0.0
    R = camera_rotation(float(pitch) if pitch is not None else 0.0, float(roll))
    return f_px, ((W - 1) / 2, (H - 1) / 2), R


def replace(hdr: np.ndarray, option: str, ref_meta: dict, ginfo: dict | None = None,
            shot_index: int = 0, seed: str | int = 0, ref_yaw: float | None = None,
            brightness: float = 1.0, prob: np.ndarray | None = None,
            name: str | None = None, chroma: float | None = None) -> tuple[np.ndarray, dict]:
    """Replace the sky of a linear HDR image (after upright) with this shot's view
    of the job's dome. Returns (hdr, info); hdr is modified in place."""
    from . import sky as skymod
    from .merge import luminance
    H, W = hdr.shape[:2]
    if prob is None:
        prob = skymod.sky_probability(hdr)
    alpha = skymod.refine_mask(prob, hdr)
    info = {"option": option, "sky_fraction": round(float(alpha.mean()), 4)}
    if alpha.mean() < 0.002:
        info["applied"] = False
        return hdr, info
    rows = np.where(alpha.max(axis=1) > 0.002)[0]
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    name = name or choose(option, seed)
    band, meta = load(name)
    f_px, pp, R = geometry_for(ginfo or {}, ref_meta, W, H)
    yaw = view_yaw(option, meta, ref_meta, shot_index, seed, ref_yaw)
    new = render_view(band, meta, W, H, f_px, pp, R, yaw, rows=(r0, r1))
    a = alpha[r0:r1]
    sel = a > 0.9
    old_Y, new_Y = luminance(hdr[r0:r1]), luminance(new)
    k = 1.0
    if sel.sum() > 100:
        k = float(np.percentile(old_Y[sel], 50) / max(np.percentile(new_Y[sel], 50), 1e-9))
    new *= np.float32(k * brightness)
    if chroma is None:  # daylight domes are a little hazier than a clean NZ sky
        chroma = 1.0 if option.startswith("twilight") else 1.15
    if chroma != 1.0:
        from .grade import oklab_to_rec2020, rec2020_to_oklab
        for y0 in range(0, new.shape[0], 512):
            lab = rec2020_to_oklab(new[y0:y0 + 512])
            lab[..., 1:] *= np.float32(chroma)
            new[y0:y0 + 512] = np.maximum(oklab_to_rec2020(lab), 0)
    # Edge decontamination: a pixel on an edge is a mix of old sky and object.
    # Swap only the old sky's share for the new sky's (exact in linear light), so
    # no halo of the old sky's colour survives around roofs, trees and posts.
    old_sky = _local_sky(hdr[r0:r1], a)
    a3 = a[..., None]
    blk = hdr[r0:r1]
    blk += a3 * (new - old_sky)
    np.maximum(blk, blk * 0 + (1 - a3) * 1e-6, out=blk)
    info["_alpha"] = alpha  # for the grade (twilight purple); not serialised
    info.update({"applied": True, "dome": name, "yaw": round(yaw, 1), "f_px": round(f_px, 1),
                 "pp": [round(pp[0], 1), round(pp[1], 1)], "exposure_match": round(k, 6)})
    return hdr, info


def _local_sky(hdr: np.ndarray, alpha: np.ndarray, scale: int = 4) -> np.ndarray:
    """Smooth estimate of the (old) sky radiance behind every pixel: the image
    itself where it is pure sky, a normalised-convolution fill of nearby sky
    pixels elsewhere."""
    H, W = alpha.shape
    h, w = max(1, H // scale), max(1, W // scale)
    small = cv2.resize(hdr, (w, h), interpolation=cv2.INTER_AREA)
    wt = (cv2.resize(alpha, (w, h), interpolation=cv2.INTER_AREA) > 0.97).astype(np.float32)
    sig = max(2.0, 0.006 * max(h, w))
    num = cv2.GaussianBlur(small * wt[..., None], (0, 0), sig)
    den = cv2.GaussianBlur(wt, (0, 0), sig)[..., None]
    fill = num / np.maximum(den, 1e-4)
    fill = np.where(den > 1e-3, fill, small)  # no sky nearby: leave the pixel alone
    fill = cv2.resize(fill, (W, H), interpolation=cv2.INTER_LINEAR)
    pure = np.clip((alpha - 0.97) / 0.03, 0, 1)[..., None]
    return hdr * pure + fill * (1 - pure)
