"""End-to-end processing: folder of bracketed RAWs -> finished JPEGs."""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import time

import cv2
import numpy as np
import rawpy

from . import denoise, geometry, merge, raw, render
from .presets import PRESETS

RAW_EXT = (".nef", ".dng", ".cr3", ".cr2", ".arw", ".raf", ".orf", ".rw2")
DNG_CONVERTER = os.environ.get(
    "KF_DNG_CONVERTER", r"C:\Program Files\Adobe\Adobe DNG Converter\Adobe DNG Converter.exe")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def find_raws(folder: str) -> list[str]:
    out = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(RAW_EXT) and not f.startswith("."):
                out.append(os.path.join(root, f))
    return sorted(out)


def needs_conversion(meta: dict) -> bool:
    """Nikon High Efficiency / High Efficiency* (TicoRAW) NEFs cannot be decoded by
    LibRaw (exiftool -n codes 13 and 14)."""
    v = meta.get("NEFCompression")
    return v in (13, 14) or "high efficiency" in str(v).lower()


def converter_available() -> bool:
    exe = os.path.expanduser("~/.wine/drive_c/Program Files/Adobe/Adobe DNG Converter/Adobe DNG Converter.exe")
    return os.path.exists(exe) and shutil.which("wine") is not None


def should_convert(meta: dict, have_converter: bool) -> bool:
    """Every camera RAW goes through Adobe DNG Converter when it is installed: the
    DNG carries Adobe's built-in lens corrections (distortion, lateral CA,
    vignetting) for Canon RF / Nikon Z bodies, so all brands get the same
    treatment. DJI files are DNG already. Without the converter only files LibRaw
    cannot read at all are attempted."""
    if str(meta.get("FileName", "")).lower().endswith(".dng"):
        return False
    return have_converter or needs_conversion(meta)


def convert_to_dng(paths: list[str], out_dir: str, jobs: int = 3) -> dict:
    """Convert RAWs with Adobe DNG Converter under Wine. Returns {src: dng}."""
    os.makedirs(out_dir, exist_ok=True)
    todo = [p for p in paths if not os.path.exists(_dng_name(p, out_dir))]
    if todo:
        wine_c = os.path.expanduser("~/.wine/drive_c")
        link_in, link_out = os.path.join(wine_c, "kf_in"), os.path.join(wine_c, "kf_out")
        for link, target in ((link_in, os.path.dirname(os.path.abspath(todo[0]))), (link_out, os.path.abspath(out_dir))):
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(target, link)
        chunks = [todo[i::jobs] for i in range(jobs)]
        procs = []
        env = dict(os.environ, WINEDEBUG="-all")
        for i, ch in enumerate(chunks):
            if not ch:
                continue
            args = ["xvfb-run", "-a", "-n", str(80 + i), "wine", DNG_CONVERTER, "-c", "-p0",
                    "-d", r"C:\kf_out"] + [r"C:\kf_in" + "\\" + os.path.basename(p) for p in ch]
            procs.append(subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env))
        for p in procs:
            p.wait()
    return {p: _dng_name(p, out_dir) for p in paths if os.path.exists(_dng_name(p, out_dir))}


def _dng_name(p: str, out_dir: str) -> str:
    return os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + ".dng")


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def merge_bracket(group: list[dict], p: dict, half: bool = False) -> tuple[np.ndarray, dict, dict]:
    """Decode, align, merge and clean one bracket -> (linear HDR, ref meta, info)."""
    t0 = time.time()
    ordered = sorted(group, key=raw.relative_exposure)
    ref_meta = ordered[len(ordered) // 2]
    with rawpy.imread(ref_meta["decode_path"]) as r:
        # night: decode at daylight balance so the fixed-CCT grade is absolute
        wb = list(r.daylight_whitebalance[:3] if p.get("wb_source") == "daylight"
                  else r.camera_whitebalance[:3])
    frames = []
    for m in ordered:
        f = raw.decode(m["decode_path"], m, m.get("nef_meta"), half=half, wb=wb,
                       nikon_vignette_strength=p.get("nikon_vignette_strength", 0.75))
        f.rgb = f.rgb.astype(np.float16)  # half the memory; ample precision for merging
        if frames:
            f.hard = None  # only the darkest frame's raw clip map is needed by the merge
        frames.append(f)
    t1 = time.time()
    merge.align(frames, len(frames) // 2)
    hdr, minfo = merge.merge(frames, deghost=p.get("deghost", True))
    del frames
    hdr = denoise.clean(hdr, p.get("chroma_nr", 1.0), p.get("luma_nr", 0.35), p.get("defringe", 1.0))
    info = {"files": [m["FileName"] for m in ordered], "merge": minfo,
            "timing": {"decode": round(t1 - t0, 1), "merge": round(time.time() - t1, 1)}}
    return hdr, ref_meta, info


def finish(hdr: np.ndarray, ref_meta: dict, p: dict, info: dict | None = None) -> tuple[np.ndarray, dict]:
    """Sky replacement, upright, signature grade and retouching -> sRGB float image."""
    info = info if info is not None else {"timing": {}}
    t0 = time.time()
    if p.get("sky"):  # a sky photo (edits.json); domes are placed after upright
        from . import sky as skymod
        hdr, info["sky"] = skymod.replace_sky(hdr, None if p["sky"] is True else p["sky"],
                                              brightness=p.get("sky_brightness", 1.0))
    is_aerial = str(ref_meta.get("Make", "")).upper().startswith("DJI")
    if is_aerial and p.get("aerial"):
        p = {**p, **p["aerial"]}  # drone shots: their own tone (see presets)
    ginfo = {}
    if p.get("upright", True) and not is_aerial:
        hdr, ginfo = geometry.upright(hdr, ref_meta, p)
    elif is_aerial and p.get("level_horizon", True):
        hdr, ginfo = geometry.level(hdr, ref_meta, p)
    if p.get("interior") and not is_aerial:
        # interiors get their own settings (twilight: tungsten-warm, not orange). An
        # interior shows almost no sky outside windows; ADE20K calls sky seen
        # through glass "windowpane", so the plain pass separates the two cleanly.
        from . import sky as skymod
        ext = float((skymod.sky_probability(hdr, windows=False) > 0.5).mean())
        info["scene"] = "exterior" if ext > p.get("exterior_min_sky", 0.05) else "interior"
        if info["scene"] == "interior":
            p = {**p, **p["interior"]}
    sky_alpha = None
    if p.get("sky_dome"):
        from . import skydome
        hdr, sinfo = skydome.replace(hdr, p["sky_dome"], ref_meta, ginfo,
                                     shot_index=p.get("shot_index", 0), seed=p.get("sky_seed", 0),
                                     ref_yaw=p.get("sky_ref_yaw"), brightness=p.get("sky_brightness", 1.0),
                                     name=p.get("sky_dome_name"))
        sky_alpha = sinfo.pop("_alpha", None)
        info["sky"] = sinfo
    if p.get("sky_purple_deg") and sky_alpha is None:
        from . import sky as skymod
        sky_alpha = skymod.refine_mask(skymod.sky_probability(hdr), hdr)
    emit = None
    if p.get("lights"):
        from . import lights
        excl = sky_alpha
        if excl is None:  # keep sky seen through windows from reading as a lit window
            from . import sky as skymod
            excl = skymod.sky_probability(hdr)
        emit = lights.emitter_map(hdr, excl)
    t1 = time.time()
    H, W = hdr.shape[:2]
    v, rinfo = render.render(hdr, p, return_info=True, inplace=True,
                             sky=sky_alpha if p.get("sky_purple_deg") else None)
    del hdr
    if p.get("sky_gradient") and sky_alpha is not None and info.get("sky", {}).get("applied"):
        from . import skydome
        f_px, pp, R = skydome.geometry_for(ginfo, ref_meta, W, H)
        v, info["sky"]["paint"] = skydome.paint_gradient(
            v, sky_alpha, f_px, pp, R, p["sky_gradient"], texture=p.get("sky_texture", 1.0),
            wisps=p.get("sky_wisps", 0.0), yaw=info["sky"].get("yaw", 0.0), seed=p.get("sky_seed", 0))
    del sky_alpha
    if emit is not None:
        v = lights.enhance(v, emit, p.get("lights_glow", 0.35), p.get("lights_pool", 0.18),
                           p.get("lights_warmth", 0.04))
        rinfo["lights_area"] = round(float(emit.mean()), 4)
    t2 = time.time()
    if p.get("retouch"):
        from . import retouch
        v, info["retouch"] = retouch.apply_edits(v, p["retouch"])
    if p.get("detect_people", True):
        from . import retouch
        cands = retouch.detect_people(v, min_score=p.get("people_min_score", 0.3))
        info["people_candidates"] = [{"score": c["score"], "box": [round(x, 4) for x in c["box"]]}
                                     for c in cands]
    info.update({"render": rinfo, "geometry": ginfo, "aerial": is_aerial})
    info["timing"].update({"geometry": round(t1 - t0, 1), "render": round(t2 - t1, 1),
                           "retouch": round(time.time() - t2, 1)})
    return v, info


def process_bracket(group: list[dict], preset: dict, overrides: dict | None = None,
                    half: bool = False) -> tuple[np.ndarray, dict]:
    p = dict(preset)
    p.update(overrides or {})
    hdr, ref_meta, info = merge_bracket(group, p, half)
    return finish(hdr, ref_meta, p, info)


def encode_jpeg(bgr: np.ndarray, quality: int = 95, max_bytes: int | None = None) -> tuple[bytes, dict]:
    """JPEG-encode, stepping quality (then chroma subsampling) down only as far as
    needed to fit `max_bytes`. Starts at q95 4:4:4."""
    steps = [(q, "444") for q in (quality, 94, 93, 92, 91, 90)] + \
            [(q, "420") for q in (95, 93, 91, 89, 87, 85, 82, 79, 75)]
    steps = [st for st in steps if st[0] <= quality] or [(quality, "444")]
    buf = None
    for q, sub in steps:
        flag = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444 if sub == "444" else cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420
        ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, flag])
        buf = enc.tobytes()
        if max_bytes is None or len(buf) <= max_bytes:
            return buf, {"quality": q, "subsampling": sub, "bytes": len(buf)}
    return buf, {"quality": steps[-1][0], "subsampling": steps[-1][1], "bytes": len(buf), "over_limit": True}


def save_jpeg(v: np.ndarray, path: str, quality: int = 95, exif_from: str | None = None,
              max_bytes: int | None = None) -> dict:
    out = (np.clip(v, 0, 1) * 255 + 0.5).astype(np.uint8)
    # leave room for the EXIF block and the sRGB ICC profile added below
    buf, enc = encode_jpeg(out[:, :, ::-1], quality, None if max_bytes is None else max_bytes - 48_000)
    with open(path, "wb") as fh:
        fh.write(buf)
    icc = os.path.join(os.path.dirname(__file__), "sRGB.icc")
    args = ["exiftool", "-q", "-overwrite_original"]
    if exif_from:
        args += ["-TagsFromFile", exif_from, "-Make", "-Model", "-LensModel", "-FocalLength",
                 "-DateTimeOriginal", "-Copyright", "-Artist"]
    if os.path.exists(icc):
        args += [f"-ICC_Profile<={icc}"]
    if len(args) > 3:
        subprocess.run(args + [path], capture_output=True)
    enc["bytes"] = os.path.getsize(path)
    return enc


def run(input_dir: str, output_dir: str, style: str = "day", half: bool = False,
        only: list[str] | None = None, web_long_edge: int = 2560, work_dir: str | None = None,
        jobs: int = 1, resume: bool = True, reverse: bool = False, options: dict | None = None,
        max_mb: float | None = None):
    """options: preset overrides for the whole shoot (presets.job_overrides plus
    e.g. sky_seed); max_mb: size cap for the full-resolution JPEGs."""
    preset = dict(PRESETS[style])
    preset.update(options or {})
    preset["_max_bytes"] = int(max_mb * 1_000_000) if max_mb else None
    work_dir = work_dir or os.path.join(output_dir, ".work")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "web"), exist_ok=True)
    files = find_raws(input_dir)
    metas = raw.exif(files)
    for m in metas:
        m["path"] = os.path.join(m["Directory"], m["FileName"])
    # brackets come from the cameras' own capture times/settings, so group first
    # and convert only what this run will edit (a re-run of two images converts six files)
    groups = merge.group_brackets(metas)

    def wanted(g):
        names = [os.path.splitext(m["FileName"])[0] for m in g]
        return not only or any(o in n for o in only for n in names)

    sel = [m for g in groups if wanted(g) for m in g]
    have_conv = converter_available()
    conv = [m["path"] for m in sel if should_convert(m, have_conv)]
    if conv:
        log(f"converting {len(conv)} camera RAWs to DNG (Adobe DNG Converter)")
        mapping = convert_to_dng(conv, os.path.join(work_dir, "dng"))
        missing = [c for c in conv if c not in mapping]
        if missing:
            log(f"WARNING: {len(missing)} files failed to convert: {[os.path.basename(x) for x in missing[:5]]}")
        dmeta = {os.path.basename(d): x for d, x in zip(mapping.values(), raw.exif(list(mapping.values())))}
        for m in sel:
            if m["path"] in mapping:
                d = mapping[m["path"]]
                dm = dmeta.get(os.path.basename(d), {})
                m["nef_meta"] = dict(m)
                m["decode_path"] = d
                # converted DNG carries the opcodes; keep original timing/exposure tags
                for k in ("ImageWidth", "ImageHeight"):
                    if k in dm:
                        m[k] = dm[k]
            elif m["path"] in missing and not needs_conversion(m):
                m["decode_path"] = m["path"]  # converter failed but LibRaw can read it: decode directly
    for m in metas:
        if "decode_path" not in m and not (m in sel and m["path"] in conv):
            m["decode_path"] = m["path"]
    # frames that could not be prepared drop out; bracket numbering stays stable
    groups = [[m for m in g if m.get("decode_path") and os.path.exists(m["decode_path"])] for g in groups]
    log(f"{len(files)} files -> {len(groups)} brackets ({style})")
    # drones carry a compass: the first drone shot sets the sky dome's reference yaw
    yaws = [m.get("GimbalYawDegree", m.get("FlightYawDegree")) for g in groups for m in g]
    yaws = [y for y in yaws if y is not None]
    if yaws and "sky_ref_yaw" not in preset:
        preset["sky_ref_yaw"] = float(yaws[0])

    edits_path = os.path.join(output_dir, "edits.json")
    edits = json.load(open(edits_path)) if os.path.exists(edits_path) else {}
    tasks = []
    for i, g in enumerate(groups, 1):
        if not g:
            continue
        names = [os.path.splitext(m["FileName"])[0] for m in sorted(g, key=raw.relative_exposure)]
        key = os.path.splitext(sorted(g, key=lambda m: merge._ts(m))[0]["FileName"])[0]
        if only and not any(o in n for o in only for n in names):
            continue
        ov = dict(edits.get(key, {}))
        ov.setdefault("shot_index", i)
        if ov.get("skip"):
            continue
        if resume and os.path.exists(os.path.join(output_dir, f"{i:02d}_{key}.jpg")):
            continue  # finished in an earlier run
        tasks.append((i, len(groups), key, g, preset, ov, half, output_dir, web_long_edge, resume))
    if reverse:
        tasks.reverse()
    report = []
    rp = os.path.join(output_dir, "report.json")
    if resume and os.path.exists(rp):
        done = {f"{t[0]:02d}_{t[2]}.jpg" for t in tasks}
        report = [r for r in json.load(open(rp)) if r.get("output") not in done and "error" not in r]
    if jobs > 1 and len(tasks) > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn")) as ex:
            for info in ex.map(_process_task, tasks):
                if info is not None:
                    report.append(info)
                    _merge_report(output_dir, info)
    else:
        for t in tasks:
            info = _process_task(t)
            if info is not None:
                report.append(info)
                _merge_report(output_dir, info)
    return report


def _merge_report(output_dir: str, info: dict):
    """Add one result to report.json, tolerating other processes writing it too."""
    rp = os.path.join(output_dir, "report.json")
    with open(rp + ".lock", "w") as lk:
        import fcntl
        fcntl.flock(lk, fcntl.LOCK_EX)
        cur = json.load(open(rp)) if os.path.exists(rp) else []
        cur = [r for r in cur if r.get("output", r.get("key")) != info.get("output", info.get("key"))]
        cur.append(info)
        _write_report(output_dir, cur)


def _claim(output_dir: str, out_name: str, skip_existing: bool = True) -> bool:
    """Claim a bracket so several processes can share one output folder. A lock
    left by a process that no longer exists is taken over."""
    if skip_existing and os.path.exists(os.path.join(output_dir, out_name)):
        return False
    ld = os.path.join(output_dir, ".locks")
    os.makedirs(ld, exist_ok=True)
    lp = os.path.join(ld, out_name + ".lock")
    for _ in range(2):
        try:
            fd = os.open(lp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                pid = int(open(lp).read().strip() or 0)
                os.kill(pid, 0)
                return False  # another live process has it
            except (ValueError, ProcessLookupError, FileNotFoundError):
                try:
                    os.remove(lp)
                except FileNotFoundError:
                    pass
    return False


def _write_report(output_dir: str, report: list):
    with open(os.path.join(output_dir, "report.json"), "w") as fh:
        json.dump(sorted(report, key=lambda r: r.get("output", r.get("key", ""))), fh, indent=1, default=str)


def _process_task(task) -> dict | None:
    i, n, key, g, preset, ov, half, output_dir, web_long_edge, resume = task
    out_name = f"{i:02d}_{key}.jpg"
    if not _claim(output_dir, out_name, skip_existing=resume):
        return None
    t = time.time()
    try:
        v, info = process_bracket(g, preset, ov, half=half)
    except Exception as e:  # keep going; report the failure
        import traceback
        log(f"[{i}/{n}] {key} FAILED: {e!r}")
        return {"key": key, "error": repr(e), "trace": traceback.format_exc()}
    p = dict(preset, **ov)
    src = sorted(g, key=raw.relative_exposure)[len(g) // 2]
    info["jpeg"] = save_jpeg(render.output_sharpen(v, p), os.path.join(output_dir, out_name), 95,
                             exif_from=src["path"], max_bytes=p.get("_max_bytes"))
    save_jpeg(render.output_sharpen(v, p, long_edge=web_long_edge),
              os.path.join(output_dir, "web", out_name), 90, exif_from=src["path"])
    info.update({"key": key, "output": out_name, "seconds": round(time.time() - t, 1)})
    log(f"[{i}/{n}] {key} -> {out_name} ({info['seconds']}s) {info['timing']}")
    try:
        os.remove(os.path.join(output_dir, ".locks", out_name + ".lock"))
    except FileNotFoundError:
        pass
    return info
