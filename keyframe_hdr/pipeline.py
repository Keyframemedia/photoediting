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
        frames.append(f)
    t1 = time.time()
    merge.align(frames, len(frames) // 2)
    hdr, minfo = merge.merge(frames, deghost=p.get("deghost", True))
    del frames
    hdr = denoise.chroma_nr(hdr, p.get("chroma_nr", 1.0))
    hdr = denoise.luma_nr(hdr, p.get("luma_nr", 0.35))
    hdr = denoise.defringe(hdr, p.get("defringe", 1.0))
    info = {"files": [m["FileName"] for m in ordered], "merge": minfo,
            "timing": {"decode": round(t1 - t0, 1), "merge": round(time.time() - t1, 1)}}
    return hdr, ref_meta, info


def finish(hdr: np.ndarray, ref_meta: dict, p: dict, info: dict | None = None) -> tuple[np.ndarray, dict]:
    """Sky replacement, upright, signature grade and retouching -> sRGB float image."""
    info = info if info is not None else {"timing": {}}
    t0 = time.time()
    if p.get("sky"):
        from . import sky as skymod
        hdr, info["sky"] = skymod.replace_sky(hdr, None if p["sky"] is True else p["sky"],
                                              brightness=p.get("sky_brightness", 1.0))
    is_aerial = str(ref_meta.get("Make", "")).upper().startswith("DJI")
    ginfo = {}
    if p.get("upright", True) and not is_aerial:
        hdr, ginfo = geometry.upright(hdr, ref_meta, p)
    elif is_aerial and p.get("level_horizon", True):
        hdr, ginfo = geometry.level(hdr, ref_meta, p)
    t1 = time.time()
    v, rinfo = render.render(hdr, p, return_info=True, inplace=True)
    del hdr
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


def save_jpeg(v: np.ndarray, path: str, quality: int = 95, exif_from: str | None = None):
    out = (np.clip(v, 0, 1) * 255 + 0.5).astype(np.uint8)
    cv2.imwrite(path, out[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality,
                                        cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
    icc = os.path.join(os.path.dirname(__file__), "sRGB.icc")
    args = ["exiftool", "-q", "-overwrite_original"]
    if exif_from:
        args += ["-TagsFromFile", exif_from, "-Make", "-Model", "-LensModel", "-FocalLength",
                 "-DateTimeOriginal", "-Copyright", "-Artist"]
    if os.path.exists(icc):
        args += [f"-ICC_Profile<={icc}"]
    if len(args) > 3:
        subprocess.run(args + [path], capture_output=True)


def run(input_dir: str, output_dir: str, style: str = "day", half: bool = False,
        only: list[str] | None = None, web_long_edge: int = 2560, work_dir: str | None = None,
        jobs: int = 1, resume: bool = True):
    preset = PRESETS[style]
    work_dir = work_dir or os.path.join(output_dir, ".work")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "web"), exist_ok=True)
    files = find_raws(input_dir)
    metas = raw.exif(files)
    for m in metas:
        m["path"] = os.path.join(m["Directory"], m["FileName"])
    conv = [m["path"] for m in metas if needs_conversion(m)]
    if conv:
        log(f"converting {len(conv)} High Efficiency NEFs to DNG")
        mapping = convert_to_dng(conv, os.path.join(work_dir, "dng"))
        missing = [c for c in conv if c not in mapping]
        if missing:
            log(f"WARNING: {len(missing)} files failed to convert: {[os.path.basename(x) for x in missing[:5]]}")
        dmeta = {os.path.basename(d): x for d, x in zip(mapping.values(), raw.exif(list(mapping.values())))}
        for m in metas:
            if m["path"] in mapping:
                d = mapping[m["path"]]
                dm = dmeta.get(os.path.basename(d), {})
                m["nef_meta"] = dict(m)
                m["decode_path"] = d
                # converted DNG carries the opcodes; keep original timing/exposure tags
                for k in ("ImageWidth", "ImageHeight"):
                    if k in dm:
                        m[k] = dm[k]
    for m in metas:
        m.setdefault("decode_path", m["path"])
    metas = [m for m in metas if os.path.exists(m["decode_path"])]
    groups = merge.group_brackets(metas)
    log(f"{len(files)} files -> {len(groups)} brackets ({style})")

    edits_path = os.path.join(output_dir, "edits.json")
    edits = json.load(open(edits_path)) if os.path.exists(edits_path) else {}
    tasks = []
    for i, g in enumerate(groups, 1):
        names = [os.path.splitext(m["FileName"])[0] for m in sorted(g, key=raw.relative_exposure)]
        key = os.path.splitext(sorted(g, key=lambda m: merge._ts(m))[0]["FileName"])[0]
        if only and not any(o in n for o in only for n in names):
            continue
        ov = edits.get(key, {})
        if ov.get("skip"):
            continue
        if resume and os.path.exists(os.path.join(output_dir, f"{i:02d}_{key}.jpg")):
            continue  # finished in an earlier run
        tasks.append((i, len(groups), key, g, preset, ov, half, output_dir, web_long_edge))
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
                report.append(info)
                _write_report(output_dir, report)
    else:
        for t in tasks:
            report.append(_process_task(t))
            _write_report(output_dir, report)
    return report


def _write_report(output_dir: str, report: list):
    with open(os.path.join(output_dir, "report.json"), "w") as fh:
        json.dump(sorted(report, key=lambda r: r.get("output", r.get("key", ""))), fh, indent=1, default=str)


def _process_task(task) -> dict:
    i, n, key, g, preset, ov, half, output_dir, web_long_edge = task
    out_name = f"{i:02d}_{key}.jpg"
    t = time.time()
    try:
        v, info = process_bracket(g, preset, ov, half=half)
    except Exception as e:  # keep going; report the failure
        import traceback
        log(f"[{i}/{n}] {key} FAILED: {e!r}")
        return {"key": key, "error": repr(e), "trace": traceback.format_exc()}
    p = dict(preset, **ov)
    src = sorted(g, key=raw.relative_exposure)[len(g) // 2]
    save_jpeg(render.output_sharpen(v, p), os.path.join(output_dir, out_name), 95, exif_from=src["path"])
    save_jpeg(render.output_sharpen(v, p, long_edge=web_long_edge),
              os.path.join(output_dir, "web", out_name), 90, exif_from=src["path"])
    info.update({"key": key, "output": out_name, "seconds": round(time.time() - t, 1)})
    log(f"[{i}/{n}] {key} -> {out_name} ({info['seconds']}s) {info['timing']}")
    return info
