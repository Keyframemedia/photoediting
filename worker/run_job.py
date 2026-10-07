"""Keyframe worker: one Photo edits job from the Keyframe portal, end to end.

The Claude session that runs a job drives these commands (see worker/WORKER.md).
Everything the portal needs - progress, Stop checks, uploads, the image list - is
sent straight to the portal's photo-edit-worker function with the job's one-time
token, so nothing has to be copied by hand.

  python3 worker/run_job.py init --job JOB_ID --token TOKEN   claim the job, set up its folder
  python3 worker/run_job.py download            Dropbox folder -> RAWs on disk
  python3 worker/run_job.py scan                count brackets (after setup)
  python3 worker/run_job.py start [--only K ...] [--force] [--workers N]   edit in the background
  python3 worker/run_job.py wait [--minutes 9]  block until progress / finish; reports progress
  python3 worker/run_job.py stop                stop the background edit
  python3 worker/run_job.py stage "TEXT"        show a stage in the portal (also a heartbeat)
  python3 worker/run_job.py review              contact sheets for the reflection check
  python3 worker/run_job.py grid KEY [--crop x0 y0 x1 y1]   one image with a labelled grid
  python3 worker/run_job.py deliver [--report TEXT]   name, upload and register the images; mark complete
  python3 worker/run_job.py fail "MESSAGE"      mark the job as needing attention
  python3 worker/run_job.py stopped             put the job back to Uploaded after a Stop
  python3 worker/run_job.py process [...]       the edit itself (what `start` runs)
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Job folders live next to the repo checkout, inside the session's working directory.
JOBS = os.environ.get("KF_JOBS") or os.path.join(os.path.dirname(ROOT), "kf_jobs")
CURRENT = os.path.join(JOBS, "current")
PORTAL = os.environ.get("KF_PORTAL_WORKER", "https://xpakdqbjgukgqbcgrkqa.supabase.co/functions/v1/photo-edit-worker")
MAX_BYTES = 10_000_000
PREVIEW_EDGE = 1600
RAW_EXT = (".nef", ".dng", ".cr3", ".cr2", ".arw", ".raf", ".orf", ".rw2")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def jdir() -> str:
    if not os.path.exists(CURRENT):
        sys.exit("no job: run `init` first")
    return os.path.realpath(CURRENT)


def load(name: str, default=None):
    p = os.path.join(jdir(), name)
    return json.load(open(p)) if os.path.exists(p) else default


def save(name: str, data) -> str:
    p = os.path.join(jdir(), name)
    with open(p + ".tmp", "w") as fh:
        json.dump(data, fh, indent=1, ensure_ascii=False)
    os.replace(p + ".tmp", p)
    return p


def safe_name(s: str) -> str:
    import re
    s = re.sub(r'[\\/:*?"<>|]+', "-", s).strip()
    return re.sub(r"\s+", " ", s)[:100] or "Keyframe"


# ---------------------------------------------------------------------------
# The portal

def portal(op: str, job: dict | None = None, **body) -> dict:
    """Call photo-edit-worker; retries transient failures."""
    job = job or load("job.json")
    payload = json.dumps({"op": op, "job_id": job["id"], "token": job["token"], **body}).encode()
    last = None
    for attempt in range(4):
        req = urllib.request.Request(PORTAL, data=payload, method="POST", headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:500]
            if e.code < 500:
                raise SystemExit(f"PORTAL REFUSED {op} ({e.code}): {msg}")
            last = f"{e.code} {msg}"
        except (urllib.error.URLError, TimeoutError) as e:
            last = str(e)
        time.sleep(2 ** attempt)
    raise SystemExit(f"PORTAL UNREACHABLE for {op}: {last}")


def push(**patch) -> dict:
    """Update the job row (always a heartbeat). Returns {cancel_requested, archived, status}."""
    r = portal("update", patch=patch)
    if r.get("cancel_requested"):
        print("CANCEL REQUESTED: the photographer pressed Stop in the portal.")
    return r


def find_raws(folder: str) -> list[str]:
    """Same rule as keyframe_hdr.pipeline.find_raws, without importing it (the
    download runs while the image libraries are still being installed)."""
    out = []
    for root, _, files in os.walk(folder):
        out += [os.path.join(root, f) for f in files if f.lower().endswith(RAW_EXT) and not f.startswith(".")]
    return sorted(out)


# ---------------------------------------------------------------------------

def cmd_init(a):
    d = os.path.join(JOBS, a.job)
    for sub in ("raw", "out", "work", "deliver", "previews", "review"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    if os.path.islink(CURRENT) or os.path.exists(CURRENT):
        os.remove(CURRENT)
    os.symlink(d, CURRENT)
    stub = {"id": a.job, "token": a.token}
    job = portal("get", stub)["job"]
    job.update(stub)
    save("job.json", job)
    if job.get("archived"):
        sys.exit("JOB IS ARCHIVED: nothing to do")
    portal("reset_images", job)  # an "Edit again" starts clean
    push(status="editing", stage="Setting up", started=True, progress_done=0, progress_total=0,
         eta=None, error=None, report=None, clear_cancel=True,
         worker_session=os.environ.get("CLAUDE_CODE_REMOTE_SESSION_ID") or None)
    print(f"claimed {job['id']}: {job['property']} [{job['style']}, sky={job['sky']}, look={job.get('look')}, "
          f"lights={job.get('lights')}]\nnotes: {job.get('notes') or '-'}\nfolder: {d}")


def _dl_url(link: str) -> str:
    link = link.strip()
    if "dl=0" in link:
        return link.replace("dl=0", "dl=1")
    if "dl=1" in link:
        return link
    return link + ("&" if "?" in link else "?") + "dl=1"


def cmd_download(a):
    job = load("job.json")
    d = jdir()
    raw = os.path.join(d, "raw")
    if find_raws(raw) and not a.force:
        print(f"already downloaded: {len(find_raws(raw))} RAW files")
        return
    push(stage="Downloading RAWs from Dropbox")
    z = os.path.join(d, "dropbox.zip")
    url = _dl_url(job["dropbox_url"])
    t = time.time()
    r = subprocess.run(["curl", "-sS", "-L", "--retry", "5", "--retry-delay", "5", "-o", z,
                        "-w", "%{http_code} %{size_download}", url], capture_output=True, text=True)
    code = r.stdout.strip().split(" ")[0] if r.stdout else "?"
    ok = r.returncode == 0 and code == "200" and os.path.exists(z)
    if ok:
        with open(z, "rb") as fh:
            magic = fh.read(4)
        if magic[:2] != b"PK" and magic[:1] == b"<":
            ok = False  # an HTML page: the link isn't public, or the folder is too big to zip
    if not ok:
        msg = ("Couldn't download the Dropbox folder. Check the link is shared as 'Anyone with the link' "
               "(and holds only the RAWs for this shoot), then press Restart editing.")
        push(status="failed", stage="Needs attention", error=msg)
        sys.exit(f"DOWNLOAD FAILED (HTTP {code}): {msg}")
    size = os.path.getsize(z)
    if magic[:2] != b"PK":
        os.replace(z, os.path.join(raw, os.path.basename(url.split("?")[0]) or "file"))
    else:
        u = subprocess.run(["unzip", "-q", "-o", z, "-d", raw, "-x", "__MACOSX/*"], capture_output=True, text=True)
        if u.returncode not in (0, 1):
            push(status="failed", stage="Needs attention", error="The Dropbox download couldn't be unpacked. Press Restart editing.")
            sys.exit(f"UNZIP FAILED: {u.stderr[:300]}")
        os.remove(z)
    n = len(find_raws(raw))
    print(f"downloaded {size / 1e9:.2f} GB in {time.time() - t:.0f}s: {n} RAW files")
    if n == 0:
        push(status="failed", stage="Needs attention", error="No RAW files (.CR3, .CR2, .NEF, .DNG) in the Dropbox folder.")
        sys.exit("NO RAW FILES")
    push(stage=f"Downloaded {n} RAW files")


def _groups():
    from keyframe_hdr import merge, raw
    files = find_raws(os.path.join(jdir(), "raw"))
    metas = raw.exif(files)
    for m in metas:
        m["path"] = os.path.join(m["Directory"], m["FileName"])
    return files, metas, merge.group_brackets(metas)


def cmd_scan(a):
    files, metas, groups = _groups()
    only = load("job.json").get("only")
    if only:
        groups = [g for g in groups if any(o in os.path.splitext(m["FileName"])[0] for o in only for m in g)]
    makes = sorted({f"{m.get('Make', '?')} {m.get('Model', '')}".strip() for m in metas})
    sizes = {}
    for g in groups:
        sizes[len(g)] = sizes.get(len(g), 0) + 1
    st = load("state.json", {})
    st.update({"files": len(files), "brackets": len(groups), "cameras": makes, "bracket_sizes": sizes, "scanned": now()})
    save("state.json", st)
    print(json.dumps(st, indent=1))
    odd = {k: v for k, v in sizes.items() if k != 3}
    if odd:
        print(f"NOTE: not every bracket has 3 frames: {odd} (single frames are edited on their own)")
    push(progress_total=len(groups), progress_done=0, stage=f"{len(groups)} brackets to edit")


def _cli_args(job: dict) -> list[str]:
    style = "day" if job.get("style") == "day" else "twilight"
    args = ["--style", style, "--sky", job.get("sky") or "original", "--seed", job["id"],
            "--max-mb", str(MAX_BYTES / 1e6), "--web-size", str(PREVIEW_EDGE),
            "--work-dir", os.path.join(jdir(), "work")]
    if style == "twilight":
        args += ["--look", job.get("look") or "natural"]
        if not job.get("lights", True):
            args.append("--no-lights")
    return args


def _outputs() -> list[str]:
    return sorted(glob.glob(os.path.join(jdir(), "out", "[0-9]*.jpg")))


def cmd_process(a):
    job = load("job.json")
    d = jdir()
    base = [sys.executable, "-m", "keyframe_hdr", os.path.join(d, "raw"), os.path.join(d, "out")] + _cli_args(job)
    only = a.only or job.get("only")  # job.only: test runs on a few brackets
    if only:
        base += ["--only", *only]
    if a.force:
        base.append("--force")
    workers = a.workers
    for attempt in range(3):
        procs = []
        for w in range(workers):
            cmd = base + (["--reverse"] if w % 2 else [])
            logf = open(os.path.join(d, f"process_{attempt}_{w}.log"), "w")
            procs.append(subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT))
            time.sleep(20)  # stagger: model loading and the first decode are the memory peaks
        codes = [p.wait() for p in procs]
        print(f"pass {attempt + 1}: exit codes {codes}", flush=True)
        if a.force or a.only or all(c == 0 for c in codes):
            break
        workers = 1  # a process was killed (usually memory): carry on one at a time
    rp = os.path.join(d, "out", "report.json")
    rep = json.load(open(rp)) if os.path.exists(rp) else []
    fails = [r for r in rep if "error" in r]
    print(f"outputs: {len(_outputs())}, failures: {len(fails)}")
    for f in fails:
        print(f"  FAILED {f.get('key')}: {f.get('error')}")
    st = load("state.json", {})
    st.update({"process_finished": now(), "failures": [f"{f.get('key')}: {f.get('error')}" for f in fails]})
    save("state.json", st)


def cmd_start(a):
    """Run `process` detached from this shell (it outlives any tool timeout)."""
    d = jdir()
    st = load("state.json", {})
    st.pop("process_finished", None)
    if not (a.only or a.force):
        st["process_started"] = now()
    save("state.json", st)
    args = [sys.executable, os.path.abspath(__file__), "process", "--workers", str(a.workers)]
    if a.only:
        args += ["--only", *a.only]
    if a.force:
        args.append("--force")
    log = open(os.path.join(d, "process.log"), "a")
    p = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    open(os.path.join(d, "process.pid"), "w").write(str(p.pid))
    push(stage="Editing" if not (a.only or a.force) else "Re-editing retouched images")
    print(f"started pid {p.pid}; log {os.path.join(d, 'process.log')}")


def _alive() -> bool:
    try:
        os.kill(int(open(os.path.join(jdir(), "process.pid")).read()), 0)
        return True
    except (OSError, ValueError):
        return False


def _progress(stage: str | None = None) -> dict:
    st = load("state.json", {})
    total = int(st.get("brackets") or 0)
    done = min(len(_outputs()), total) if total else len(_outputs())
    patch = {"progress_done": done, "progress_total": total,
             "stage": stage or (f"Editing {done} of {total}" if total else "Editing"), "eta": None}
    if done and st.get("process_started") and done < total:
        t0 = dt.datetime.strptime(st["process_started"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
        el = (dt.datetime.now(dt.timezone.utc) - t0).total_seconds()
        left = (total - done) * el / done
        patch["eta"] = f"{max(1, round(left / 60))} min" if left < 5400 else f"{left / 3600:.1f} h"
    return patch


def cmd_wait(a):
    """Block until the edit finishes, `--images` more are done or `--minutes` pass;
    send progress to the portal; print one status line."""
    t0 = time.time()
    start = len(_outputs())
    while True:
        st = load("state.json", {})
        done = len(_outputs())
        if st.get("process_finished"):
            push(**_progress("Edited, checking"))
            print(f"FINISHED {done}/{st.get('brackets')} images; failures: {len(st.get('failures', []))}")
            for f in st.get("failures", []):
                print("  FAILED", f)
            return
        if not _alive():
            tail = open(os.path.join(jdir(), "process.log")).read()[-1500:] if os.path.exists(os.path.join(jdir(), "process.log")) else ""
            print(f"DIED {done}/{st.get('brackets')} images - the edit process exited without finishing. Log tail:\n{tail}")
            return
        if done - start >= a.images or time.time() - t0 > a.minutes * 60:
            r = push(**_progress())
            print(("CANCELLED " if r.get("cancel_requested") else "RUNNING ") + f"{done}/{st.get('brackets')} images")
            return
        time.sleep(15)


def cmd_stop(a):
    try:
        pid = int(open(os.path.join(jdir(), "process.pid")).read())
        os.killpg(pid, 15)
        print(f"stopped {pid}")
    except (OSError, ValueError) as e:
        print(f"nothing to stop ({e})")


def cmd_stage(a):
    r = push(stage=a.text)
    print(json.dumps(r))


def cmd_fail(a):
    push(status="failed", stage="Needs attention", error=a.message)
    print("marked as needing attention")


def cmd_stopped(a):
    push(status="uploaded", stage="Stopped", clear_cancel=True, eta=None)
    print("back to Uploaded")


def cmd_review(a):
    """Contact sheets (6 per sheet) of every finished image with the detector's
    people candidates boxed and numbered, for the reflection check."""
    import cv2
    d = jdir()
    rep = {r.get("output"): r for r in json.load(open(os.path.join(d, "out", "report.json")))} \
        if os.path.exists(os.path.join(d, "out", "report.json")) else {}
    webs = sorted(glob.glob(os.path.join(d, "out", "web", "[0-9]*.jpg")))
    for f in glob.glob(os.path.join(d, "review", "sheet_*.jpg")):
        os.remove(f)
    tiles, lines = [], []
    for f in webs:
        name = os.path.basename(f)
        im = cv2.imread(f)
        h, w = im.shape[:2]
        s = 900 / w
        im = cv2.resize(im, (900, int(h * s)), interpolation=cv2.INTER_AREA)
        cands = rep.get(name, {}).get("people_candidates", [])
        for k, c in enumerate(cands):
            x0, y0, x1, y1 = c["box"]
            p0 = (int(x0 * 900), int(y0 * h * s))
            p1 = (int(x1 * 900), int(y1 * h * s))
            cv2.rectangle(im, p0, p1, (0, 0, 255), 2)
            cv2.putText(im, f"P{k} {c['score']:.2f}", (p0[0], max(14, p0[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.rectangle(im, (0, 0), (900, 26), (0, 0, 0), -1)
        cv2.putText(im, name[:-4], (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(im)
        if cands:
            lines.append(f"{name[:-4]}: " + ", ".join(f"P{k} score {c['score']} box {c['box']}" for k, c in enumerate(cands)))
    n = 0
    for i in range(0, len(tiles), 6):
        grp = tiles[i:i + 6]
        rows = []
        for r in range(0, len(grp), 2):
            pair = grp[r:r + 2]
            hh = max(t.shape[0] for t in pair)
            pair = [cv2.copyMakeBorder(t, 0, hh - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in pair]
            if len(pair) == 1:
                pair.append(pair[0] * 0)
            rows.append(cv2.hconcat(pair))
        n += 1
        out = os.path.join(d, "review", f"sheet_{n:02d}.jpg")
        cv2.imwrite(out, cv2.vconcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(out)
    print("\nPeople the detector flagged (check each: a real person/photographer in a reflection or "
          "shadow gets removed; people in artwork, photos or on TV screens stay):")
    print("\n".join(lines) or "  none")


def cmd_grid(a):
    """One finished image (web size) with a labelled 10x10 grid: x and y in
    tenths, so a retouch box can be read off as fractions."""
    import cv2
    d = jdir()
    f = [x for x in glob.glob(os.path.join(d, "out", "web", "[0-9]*.jpg")) if a.key in os.path.basename(x)]
    if not f:
        sys.exit(f"no image matching {a.key}")
    im = cv2.imread(f[0])
    if a.crop:
        x0, y0, x1, y1 = a.crop
        H, W = im.shape[:2]
        im = im[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]
        ox, oy, sx, sy = x0, y0, x1 - x0, y1 - y0
    else:
        ox, oy, sx, sy = 0, 0, 1, 1
    s = 1500 / im.shape[1]
    im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    H, W = im.shape[:2]
    for k in range(1, 10):
        x, y = int(W * k / 10), int(H * k / 10)
        cv2.line(im, (x, 0), (x, H), (0, 255, 255), 1)
        cv2.line(im, (0, y), (W, y), (0, 255, 255), 1)
        cv2.putText(im, f"{ox + sx * k / 10:.2f}", (x + 3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(im, f"{oy + sy * k / 10:.2f}", (3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    out = os.path.join(d, "review", f"grid_{a.key}.jpg")
    cv2.imwrite(out, im, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(out)


def cmd_deliver(a):
    """Final names (Property - 01.jpg), size check, previews, uploads, the image list,
    and status Edit complete."""
    import cv2
    sys.path.insert(0, ROOT)
    from keyframe_hdr.pipeline import encode_jpeg
    d = jdir()
    job = load("job.json")
    prop = safe_name(job.get("property", "Keyframe"))
    for sub in ("deliver", "previews"):
        for f in glob.glob(os.path.join(d, sub, "*.jpg")):
            os.remove(f)
    outs = _outputs()
    if not outs:
        push(status="failed", stage="Needs attention", error="The edit produced no images. Press Restart editing.")
        sys.exit("NO IMAGES")
    push(stage="Uploading images", progress_done=len(outs), progress_total=len(outs), eta=None)
    images = []
    for n, f in enumerate(outs, 1):
        name = f"{prop} - {n:02d}.jpg"
        dst = os.path.join(d, "deliver", name)
        shutil.copyfile(f, dst)
        if os.path.getsize(dst) > MAX_BYTES:  # belt and braces: the pipeline already caps it
            buf, _ = encode_jpeg(cv2.imread(f), 92, MAX_BYTES - 64_000)
            open(dst, "wb").write(buf)
            subprocess.run(["exiftool", "-q", "-overwrite_original", "-TagsFromFile", f, "-all:all", "-ICC_Profile", dst],
                           capture_output=True)
        web = os.path.join(d, "out", "web", os.path.basename(f))
        im = cv2.imread(web if os.path.exists(web) else f)
        h, w = im.shape[:2]
        if max(h, w) > PREVIEW_EDGE:
            s = PREVIEW_EDGE / max(h, w)
            im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        pv = os.path.join(d, "previews", name)
        cv2.imwrite(pv, im, [cv2.IMWRITE_JPEG_QUALITY, 84])
        r = subprocess.run(["exiftool", "-s3", "-ImageWidth", "-ImageHeight", dst], capture_output=True, text=True)
        try:
            ww, hh = [int(x) for x in r.stdout.split()]
        except ValueError:
            ww = hh = None
        images.append({"n": n, "file_name": name, "bytes": os.path.getsize(dst), "width": ww, "height": hh,
                       "_full": dst, "_preview": pv})
    # signed upload URLs in batches, then PUT each file
    files = [(im, "full") for im in images] + [(im, "preview") for im in images]
    for k in range(0, len(files), 50):
        chunk = files[k:k + 50]
        # storage keys must be plain ASCII ("Wānaka" isn't): NN.jpg; the real name lives in file_name
        ups = portal("upload_urls", files=[{"name": f"{im['n']:03d}.jpg", "kind": kind} for im, kind in chunk])["uploads"]
        for (im, kind), up in zip(chunk, ups):
            path = im["_full"] if kind == "full" else im["_preview"]
            for attempt in range(4):
                r = subprocess.run(["curl", "-sS", "-X", "PUT", "-H", "Content-Type: image/jpeg", "-H", "x-upsert: true",
                                    "--data-binary", "@" + path, "-o", "/dev/null", "-w", "%{http_code}", up["url"]],
                                   capture_output=True, text=True)
                if r.stdout.strip() in ("200", "201"):
                    break
                time.sleep(2 ** attempt)
            else:
                push(status="failed", stage="Needs attention", error=f"Upload of {im['file_name']} failed. Press Restart editing.")
                sys.exit(f"UPLOAD FAILED {im['file_name']} ({kind}): HTTP {r.stdout} {r.stderr[:200]}")
            im["full_path" if kind == "full" else "preview_path"] = up["path"]
        push(stage=f"Uploading images ({min(k + 50, len(files))} of {len(files)} files)")
    rows = [{k: v for k, v in im.items() if not k.startswith("_")} for im in images]
    portal("add_images", images=rows)
    report = a.report or None
    st = load("state.json", {})
    if st.get("failures"):
        extra = f"{len(st['failures'])} bracket(s) couldn't be edited: " + "; ".join(st["failures"])[:400]
        report = f"{report} {extra}" if report else extra
    push(status="complete", stage="Edit complete", finished=True, report=report, eta=None, error=None,
         progress_done=len(images), progress_total=len(images))
    total = sum(im["bytes"] for im in images)
    print(f"delivered {len(images)} images, {total / 1e9:.2f} GB, largest {max(im['bytes'] for im in images) / 1e6:.2f} MB")


def main():
    sys.path.insert(0, ROOT)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init"); s.add_argument("--job", required=True); s.add_argument("--token", required=True)
    s = sub.add_parser("download"); s.add_argument("--force", action="store_true")
    sub.add_parser("scan")
    for name in ("process", "start"):
        s = sub.add_parser(name); s.add_argument("--only", nargs="*"); s.add_argument("--force", action="store_true")
        s.add_argument("--workers", type=int, default=2)
    s = sub.add_parser("wait"); s.add_argument("--minutes", type=float, default=9); s.add_argument("--images", type=int, default=10**6)
    sub.add_parser("stop")
    s = sub.add_parser("stage"); s.add_argument("text")
    sub.add_parser("review")
    s = sub.add_parser("grid"); s.add_argument("key"); s.add_argument("--crop", nargs=4, type=float)
    s = sub.add_parser("deliver"); s.add_argument("--report")
    s = sub.add_parser("fail"); s.add_argument("message")
    sub.add_parser("stopped")
    a = ap.parse_args()
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
