"""The Keyframe editing farm: edit one Photo edits job from the portal, end to end,
with no person or AI in the loop.

    run(job_id, token, keys, map_fn)
        reads the job, finds the brackets in its Dropbox folder (reading only the
        first 2 MB of each RAW for the capture data), then edits every bracket with
        map_fn(edit_bracket, specs) - on Modal that is one machine per bracket, so a
        shoot takes about as long as its slowest photo - reporting progress, an ETA
        and honouring Stop. keys: re-edit only those photos (after review).

    edit_bracket(spec)
        downloads one bracket's RAWs, converts Nikon HE / Canon files to DNG, edits
        it in the house look with the job's options, removes people found in
        glass/mirrors (flags uncertain ones), saves the full-size JPEG (under 10 MB)
        and a preview, uploads both to the portal and the full one to the Dropbox
        folder's "Edited" folder.

Talks to the portal only through photo-edit-worker with the job's own token.

Local run (for testing; edits on this machine with a process pool):
    python3 worker/farm.py --job <id> --token <token> [--workers 2] [--keys DSC_0001 ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PORTAL = os.environ.get("KF_PORTAL_WORKER", "https://xpakdqbjgukgqbcgrkqa.supabase.co/functions/v1/photo-edit-worker")
MAX_BYTES = 10_000_000
PREVIEW_EDGE = 1600
HEAD_BYTES = 2 * 1024 * 1024


class Refused(Exception):
    """The portal no longer accepts this token (the job was restarted or deleted)."""


class Cancelled(Exception):
    """Stop was pressed."""


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Portal:
    def __init__(self, job_id: str, token: str):
        self.job_id, self.token = job_id, token

    def call(self, op: str, **body) -> dict:
        data = json.dumps({"job_id": self.job_id, "token": self.token, "op": op, **body}).encode()
        last = None
        for attempt in range(6):
            req = urllib.request.Request(PORTAL, data=data, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=90) as r:
                    return json.loads(r.read().decode() or "{}")
            except urllib.error.HTTPError as e:
                msg = e.read().decode()[:400]
                if e.code == 403:
                    raise Refused(msg)
                if 400 <= e.code < 500:
                    raise RuntimeError(f"portal {op}: HTTP {e.code} {msg}")
                last = f"HTTP {e.code} {msg}"
            except Exception as e:  # network
                last = repr(e)
            time.sleep(min(30, 2 ** attempt))
        raise RuntimeError(f"portal {op} failed: {last}")

    def update(self, **patch) -> dict:
        return self.call("update", patch=patch)


# ---------------------------------------------------------------------------
# Job settings -> pipeline options
# ---------------------------------------------------------------------------

def safe_name(s: str) -> str:
    s = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', " ", s or "").strip(" .")
    return re.sub(r"\s+", " ", s)[:120] or "Keyframe"


def job_options(job: dict) -> tuple[str, dict]:
    """(style, overrides) for the whole shoot, from what was ticked in the portal."""
    from keyframe_hdr import presets
    style = "twilight" if job.get("style") == "twilight" else "day"
    sky = job.get("sky") or "original"
    opts = job.get("options") or {}
    ov = presets.job_overrides(style, "original" if sky == "library" else sky,
                               job.get("look") or "natural", bool(job.get("lights", True)))
    ov["sky_seed"] = job["id"]
    ov["auto_people"] = bool(opts.get("remove_people", True))
    ov["tv_black"] = bool(opts.get("tv_black", False))
    ov["green_lawn"] = bool(opts.get("green_lawn", False)) and style == "day"
    if opts.get("window_pull") == "strong":
        ov.update(presets.STRONG_WINDOW_PULL)
    return style, ov


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------

def download(url: str, path: str, head: int | None = None, tries: int = 5) -> int:
    hdrs = {"User-Agent": "keyframe-farm/1"}
    if head:
        hdrs["Range"] = f"bytes=0-{head - 1}"
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=120) as r, open(path + ".part", "wb") as fh:
                ctype = r.headers.get("Content-Type", "")
                first = r.read(1 << 16)
                if first[:1] == b"<" and "html" in ctype:
                    raise RuntimeError("got a web page instead of the file (is the link shared as 'anyone with the link'?)")
                fh.write(first)
                n = len(first)
                while True:
                    if head and n >= head:
                        break
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    n += len(chunk)
            os.replace(path + ".part", path)
            return n
        except Exception as e:
            last = e
            time.sleep(min(20, 2 ** attempt))
    raise RuntimeError(f"download failed for {os.path.basename(path)}: {last}")


def upload_put(url: str, path: str, content_type: str = "image/jpeg", method: str = "PUT", tries: int = 5):
    data = open(path, "rb").read()
    last = None
    for attempt in range(tries):
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": content_type, "x-upsert": "true"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                if r.status in (200, 201):
                    return
                last = f"HTTP {r.status}"
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code} {e.read().decode()[:200]}"
        except Exception as e:
            last = repr(e)
        time.sleep(min(20, 2 ** attempt))
    raise RuntimeError(f"upload failed: {last}")


# ---------------------------------------------------------------------------
# Brackets
# ---------------------------------------------------------------------------

def find_brackets(files: list[dict], work: str) -> list[dict]:
    """Read the start of every RAW (capture time, settings) and group them into
    brackets, in shooting order. Each: {n, key, files: [file dicts]}."""
    from keyframe_hdr import merge, raw
    heads = os.path.join(work, "heads")
    os.makedirs(heads, exist_ok=True)
    by_name = {}
    for f in files:
        by_name[f["name"]] = f

    def head(f):
        p = os.path.join(heads, f["name"])
        download(f["url"], p, head=HEAD_BYTES)
        return p

    with ThreadPoolExecutor(16) as ex:
        paths = list(ex.map(head, files))
    metas = raw.exif(paths)
    missing = [m for m in metas if not m.get("DateTimeOriginal")]
    if missing:  # some formats keep their capture data further in: read those whole
        for m in missing:
            download(by_name[m["FileName"]]["url"], os.path.join(heads, m["FileName"]))
        metas = raw.exif(paths)
    groups = merge.group_brackets(metas)
    out = []
    for i, g in enumerate(groups, 1):
        g = sorted(g, key=merge._ts)
        key = os.path.splitext(g[0]["FileName"])[0]
        out.append({"n": i, "key": key, "files": [by_name[m["FileName"]] for m in g]})
    return out


def edit_bracket(spec: dict) -> dict:
    """Edit one bracket end to end (see module docstring). Returns the image row
    for the portal, or {"n", "key", "error"}."""
    import cv2
    import numpy as np
    from keyframe_hdr import pipeline, presets, raw, render

    t0 = time.time()
    portal = Portal(spec["job_id"], spec["token"])
    work = tempfile.mkdtemp(prefix=f"kf_{spec['key']}_")
    try:
        rawdir = os.path.join(work, "raw")
        os.makedirs(rawdir)
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(lambda f: download(f["url"], os.path.join(rawdir, f["name"])), spec["files"]))
        t_dl = time.time() - t0
        paths = [os.path.join(rawdir, f["name"]) for f in spec["files"]]
        metas = raw.exif(paths)
        for m in metas:
            m["path"] = os.path.join(m["Directory"], m["FileName"])
        group = pipeline.prepare_group(metas, os.path.join(work, "dng"))
        if not group:
            raise RuntimeError("none of this bracket's files could be read")

        p = dict(presets.PRESETS[spec["style"]])
        p.update(spec["overrides"])
        p["shot_index"] = spec["n"]
        if spec.get("sky_ref_yaw") is not None:
            p["sky_ref_yaw"] = spec["sky_ref_yaw"]
        if spec.get("sky_photo_url"):
            sp = os.path.join(work, "sky" + os.path.splitext(spec["sky_photo_url"].split("?")[0])[1][:5])
            download(spec["sky_photo_url"], sp)
            p["sky"] = sp
            p["sky_offset"] = spec.get("sky_offset")
            p["sky_dome"] = None
            p["sky_gradient"] = None
        edits = spec.get("edits") or {}
        p["keep_boxes"] = edits.get("keep", [])
        retouch = [{"box": b, "mode": "person"} for b in edits.get("remove", [])]
        retouch += [{"poly": poly} for poly in edits.get("brush", [])]
        if retouch:
            p["retouch"] = retouch

        v, info = pipeline.process_bracket(group, p, {}, half=bool(spec.get("half")))
        src = sorted(group, key=raw.relative_exposure)[len(group) // 2]
        name = f"{spec['property']} - {spec['n']:02d}.jpg"
        full = os.path.join(work, name)
        enc = pipeline.save_jpeg(render.output_sharpen(v, p), full, 95, exif_from=src["path"], max_bytes=MAX_BYTES)
        prev = os.path.join(work, "preview.jpg")
        pv = render.output_sharpen(v, p, long_edge=PREVIEW_EDGE)
        cv2.imwrite(prev, (np.clip(pv, 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 84])
        h, w = v.shape[:2]
        del v, pv

        # portal storage: plain ASCII keys (names like "Wānaka" aren't valid keys)
        key_name = f"{spec['n']:03d}.jpg"
        ups = portal.call("upload_urls", files=[{"name": key_name, "kind": "full"},
                                                {"name": key_name, "kind": "preview"}])["uploads"]
        upload_put(ups[0]["url"], full)
        upload_put(ups[1]["url"], prev)

        # Dropbox "Edited" folder, best effort
        dropbox = None
        try:
            d = portal.call("dropbox_upload_links", names=[name])
            if d.get("available"):
                upload_put(d["links"][0]["url"], full, content_type="application/octet-stream", method="POST")
                dropbox = d.get("folder")
        except Refused:
            raise
        except Exception as e:
            log(f"[{spec['n']}] Dropbox copy failed: {e}")

        flags = info.get("people", [])
        pending = any(f["action"] == "review" for f in flags)
        return {"n": spec["n"], "key": spec["key"], "file_name": name, "full_path": ups[0]["path"],
                "preview_path": ups[1]["path"], "bytes": enc["bytes"], "width": w, "height": h,
                "flags": flags, "review": "pending" if pending else "done" if flags else "none",
                "dropbox": dropbox,
                "seconds": round(time.time() - t0, 1), "download_s": round(t_dl, 1),
                "timing": info.get("timing", {})}
    except Refused:
        raise
    except Exception as e:
        import traceback
        log(f"[{spec['n']}] {spec['key']} FAILED: {e!r}\n{traceback.format_exc()}")
        return {"n": spec["n"], "key": spec["key"], "error": str(e)[:300]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _eta(done: int, total: int, started: float, per_photo: list[float], parallel: bool) -> str | None:
    if done >= total:
        return None
    left = total - done
    if parallel and per_photo:
        # everything runs at once: the rest finish around the slowest photo's time
        secs = max(60.0, max(per_photo) * 1.1 - (time.time() - started))
    elif per_photo:
        secs = sum(per_photo) / len(per_photo) * left
    else:
        return None
    return f"about {max(1, round(secs / 60))} min"


def run(job_id: str, token: str, keys: list[str] | None, map_fn, parallel: bool = True) -> dict:
    """Edit the whole job (or only `keys`). map_fn(fn, specs) yields results as they
    finish. Returns a summary."""
    portal = Portal(job_id, token)
    t0 = time.time()
    try:
        g = portal.call("get")
    except Refused:
        log("token refused: the job was restarted or deleted")
        return {"refused": True}
    job, images, skies = g["job"], g.get("images", []), g.get("skies", [])
    if job.get("archived"):
        return {"archived": True}
    reedit = bool(keys)
    portal.update(status="editing", stage="Finding the brackets" if not reedit else f"Re-editing {len(keys)} photos",
                  started=not reedit, clear_cancel=True, error=None, eta=None)

    stop = threading.Event()

    def heartbeat():
        while not stop.wait(60):
            try:
                r = portal.update()
                if r.get("cancel_requested"):
                    stop.set()
            except Refused:
                stop.set()
            except Exception:
                pass

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()
    work = tempfile.mkdtemp(prefix="kf_job_")
    results = None
    try:
        src = portal.call("source_files")
        files = src.get("files", [])
        if not files:
            portal.update(status="failed", stage="Needs attention", finished=True,
                          error="No RAW files (.CR3, .CR2, .NEF, .ARW, .DNG) in the Dropbox folder, or the link isn't shared.")
            return {"error": "no files"}
        brackets = find_brackets(files, work)
        style, ov = job_options(job)
        # drones carry a compass: the first drone shot sets the sky dome's direction
        from keyframe_hdr import raw
        heads = [os.path.join(work, "heads", b["files"][0]["name"]) for b in brackets]
        yaws = [m.get("GimbalYawDegree", m.get("FlightYawDegree")) for m in raw.exif(heads)]
        yaws = [float(y) for y in yaws if y is not None]
        prop = safe_name(job.get("property"))
        by_key = {im["key"]: im for im in images if im.get("key")}
        if reedit:
            brackets = [b for b in brackets if b["key"] in keys]
            for b in brackets:  # keep each photo's number
                if b["key"] in by_key:
                    b["n"] = by_key[b["key"]]["n"]
        else:
            portal.call("reset_images")
        sky_urls = [s["url"] for s in skies if s.get("url")]
        pick = int(hashlib.sha1(job["id"].encode()).hexdigest(), 16)
        specs = []
        for b in brackets:
            spec = {"job_id": job_id, "token": token, "n": b["n"], "key": b["key"], "files": b["files"],
                    "style": style, "overrides": ov, "property": prop,
                    "sky_ref_yaw": yaws[0] if yaws else None,
                    "edits": (by_key.get(b["key"]) or {}).get("edits") if reedit else None}
            if sky_urls:  # one library sky per shoot, seen from a different part per photo
                spec["sky_photo_url"] = sky_urls[pick % len(sky_urls)]
                spec["sky_offset"] = ((b["n"] * 0.381966) + (pick % 1000) / 1000) % 1.0
            specs.append(spec)
        total = len(specs)
        portal.update(progress_total=total, progress_done=0,
                      stage=f"Editing {total} photo{'s' if total != 1 else ''}", eta=None)
        log(f"{total} brackets, style {style}, {len(files)} files, {time.time() - t0:.0f}s to plan")

        done, rows, failures, per_photo = 0, [], [], []
        t_edit = time.time()
        results = map_fn(edit_bracket, specs)
        for res in results:
            if stop.is_set():
                raise Cancelled()
            if isinstance(res, BaseException):
                failures.append(f"photo: {res}")
                continue
            done += 1
            if res.get("error"):
                failures.append(f"{res['key']}: {res['error']}")
            else:
                rows.append(res)
                per_photo.append(res.get("seconds", 0))
            r = portal.update(progress_done=done, eta=_eta(done, total, t_edit, per_photo, parallel),
                              stage=f"Edited {done} of {total}")
            if r.get("cancel_requested"):
                raise Cancelled()
        if rows:
            keep = ("n", "key", "file_name", "full_path", "preview_path", "bytes", "width", "height", "flags", "review")
            portal.call("add_images", images=[{k: r[k] for k in keep} for r in sorted(rows, key=lambda r: r["n"])])
        if not rows:
            portal.update(status="failed", stage="Needs attention", finished=True,
                          error=("No photo could be edited. " + "; ".join(failures))[:2000])
            return {"error": "nothing edited", "failures": failures}
        removed = sum(1 for r in rows for f in r["flags"] if f["action"] == "remove")
        review = sum(1 for r in rows if r["review"] == "pending")
        parts = []
        if removed:
            parts.append(f"Removed {removed} reflection{'s' if removed != 1 else ''} of people.")
        if review:
            parts.append(f"{review} photo{'s' if review != 1 else ''} to check for people.")
        if failures:
            parts.append(f"{len(failures)} couldn't be edited: " + "; ".join(failures)[:600])
        dropbox = next((r["dropbox"] for r in rows if r.get("dropbox")), None)
        mins = (time.time() - t0) / 60
        parts.append(f"Edited in {mins:.0f} min.")
        portal.update(status="complete", stage="Edit complete", finished=True, eta=None, error=None,
                      report=" ".join(parts)[:2000], dropbox_folder=dropbox,
                      progress_done=total, progress_total=total)
        log(f"done: {len(rows)} photos, {len(failures)} failed, {mins:.1f} min")
        return {"photos": len(rows), "failures": failures, "minutes": round(mins, 1)}
    except Cancelled:
        if results is not None and hasattr(results, "close"):
            results.close()  # cancels the photos still being edited
        portal.update(status="uploaded", stage="Stopped", eta=None, clear_cancel=True)
        log("stopped")
        return {"cancelled": True}
    except Refused:
        log("token refused mid-run: a newer run owns the job")
        return {"refused": True}
    except Exception as e:
        import traceback
        log(f"job failed: {e!r}\n{traceback.format_exc()}")
        try:
            portal.update(status="failed", stage="Needs attention", eta=None,
                          error=f"The edit stopped: {str(e)[:300]}. Press Restart editing.")
        except Exception:
            pass
        return {"error": str(e)}
    finally:
        stop.set()
        shutil.rmtree(work, ignore_errors=True)


def local_map(workers: int):
    """map_fn running brackets on this machine (process pool), in completion order."""
    def m(fn, specs):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            futs = [ex.submit(fn, s) for s in specs]
            for f in as_completed(futs):
                try:
                    yield f.result()
                except Exception as e:
                    yield e
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--keys", nargs="*")
    a = ap.parse_args()
    print(json.dumps(run(a.job, a.token, a.keys or None, local_map(a.workers), parallel=a.workers > 1)))
