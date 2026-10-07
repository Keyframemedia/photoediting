"""Build the client-review gallery page for a processed shoot.

    python tools/make_gallery.py <output_dir> <raw_dir> <gallery_dir> --title "Wānaka" ...

Writes <gallery_dir>/index.html plus after/ and before/ image folders, ready to
publish as an Artifact (the page declares the `db` capability so each image can
be marked Approved / Needs changes with a note, which Claude reads back).
"""
from __future__ import annotations

import argparse
import html
import json
import os
import subprocess

from PIL import Image


def exif_line(raw_dir: str, files: list[str]) -> dict:
    paths = [os.path.join(raw_dir, f.replace(".dng", ".NEF") if not f.startswith("DJI") else f) for f in files]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        return {}
    out = subprocess.run(["exiftool", "-j", "-ExposureTime", "-FNumber", "-ISO", "-FocalLength",
                          "-LensModel", "-Model", *paths], capture_output=True, text=True).stdout
    rows = json.loads(out) if out.strip() else []
    if not rows:
        return {}
    shutters = " · ".join(str(r.get("ExposureTime", "")) for r in rows)
    r0 = rows[0]
    return {"shutters": shutters, "f": r0.get("FNumber"), "iso": r0.get("ISO"),
            "focal": str(r0.get("FocalLength", "")).replace(".0 mm", " mm"), "lens": r0.get("LensModel", ""),
            "model": r0.get("Model", "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("raw_dir")
    ap.add_argument("gallery_dir")
    ap.add_argument("--thumbs", help="folder of camera preview JPEGs named <stem>.jpg (the 'before')")
    ap.add_argument("--title", default="Shoot Review")
    ap.add_argument("--heading", default="Daytime test shoot")
    ap.add_argument("--meta", default="")
    ap.add_argument("--after-size", type=int, default=2200)
    a = ap.parse_args()

    report = json.load(open(os.path.join(a.output_dir, "report.json")))
    report = [r for r in report if "output" in r]
    report.sort(key=lambda r: r["output"])
    os.makedirs(os.path.join(a.gallery_dir, "after"), exist_ok=True)
    os.makedirs(os.path.join(a.gallery_dir, "before"), exist_ok=True)
    items = []
    for r in report:
        key, out = r["key"], r["output"]
        src = os.path.join(a.output_dir, "web", out)
        im = Image.open(src)
        im.thumbnail((a.after_size, a.after_size), Image.LANCZOS)
        im.save(os.path.join(a.gallery_dir, "after", out), quality=86, optimize=True, progressive=True)
        mid = r["files"][len(r["files"]) // 2].rsplit(".", 1)[0]
        before = None
        if a.thumbs and os.path.exists(os.path.join(a.thumbs, mid + ".jpg")):
            b = Image.open(os.path.join(a.thumbs, mid + ".jpg"))
            b.thumbnail((1600, 1600), Image.LANCZOS)
            before = f"before/{mid}.jpg"
            b.save(os.path.join(a.gallery_dir, before), quality=82, optimize=True, progressive=True)
        g = r.get("geometry") or {}
        notes = []
        if g.get("applied"):
            notes.append(f"Verticals corrected {g.get('tilt_deg', 0):.1f}°")
        if r.get("aerial"):
            notes.append("Aerial")
        for e in r.get("retouch") or []:
            label = {"neutral": "Moiré removed", "fill": "Object removed"}.get(
                e.get("mode"), "Photographer shadow removed" if "poly" in e else "Photographer removed from reflection")
            if label not in notes:
                notes.append(label)
        if r.get("sky", {}).get("applied"):
            notes.append("Sky replaced")
        ex = exif_line(a.raw_dir, r["files"])
        items.append({"key": key, "n": out.split("_")[0], "after": f"after/{out}", "before": before,
                      "notes": notes, "exif": ex,
                      "w": im.size[0], "h": im.size[1]})
    retouched = sum(1 for i in items if any("removed" in n for n in i["notes"]))
    page = TEMPLATE.replace("__TITLE__", html.escape(a.title)).replace("__HEADING__", html.escape(a.heading))
    page = page.replace("__META__", html.escape(a.meta)).replace("__COUNT__", str(len(items)))
    page = page.replace("__RETOUCHED__", str(retouched))
    page = page.replace("__DATA__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/"))
    with open(os.path.join(a.gallery_dir, "index.html"), "w") as fh:
        fh.write(page)
    print(f"{len(items)} images -> {a.gallery_dir}")


TEMPLATE = r"""<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;1,500&family=Inter:wght@400;500;600&display=swap">
<style>
/* Layout: editorial contact sheet - a quiet masthead, then a responsive grid of
   finished frames; each opens into a full-width before/after compare. */
:root {
  --paper: #f6f6f4; --sheet: #ffffff; --ink: #151515; --muted: #6a6a66; --line: #dddcd7;
  --ok: #2d6a3e; --ok-bg: #e5efe7; --flag: #8f4f0c; --flag-bg: #f6eadb; --focus: #151515;
  --display: "Cormorant Garamond", "Iowan Old Style", Georgia, serif;
  --body: "Inter", system-ui, -apple-system, "Segoe UI", sans-serif;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --paper: #111111; --sheet: #1a1a1a; --ink: #ecebe7; --muted: #9b9a95; --line: #2e2e2c;
  --ok: #8fcb9d; --ok-bg: #1d2e22; --flag: #e8b273; --flag-bg: #33261a; --focus: #ecebe7; color-scheme: dark } }
:root[data-theme="dark"] {
  --paper: #111111; --sheet: #1a1a1a; --ink: #ecebe7; --muted: #9b9a95; --line: #2e2e2c;
  --ok: #8fcb9d; --ok-bg: #1d2e22; --flag: #e8b273; --flag-bg: #33261a; --focus: #ecebe7; color-scheme: dark }
* { box-sizing: border-box }
[hidden] { display: none !important }
body { background: var(--paper); color: var(--ink); font: 15px/1.55 var(--body); padding: 0 clamp(16px, 4vw, 48px) 64px; }
.wrap { max-width: 1480px; margin: 0 auto }
header { padding-block: 40px 28px; border-bottom: 1px solid var(--line); display: grid; gap: 14px }
.brand { font: 600 11px/1 var(--body); letter-spacing: .22em; text-transform: uppercase; color: var(--muted) }
h1 { font: 500 clamp(34px, 5vw, 58px)/1.02 var(--display); margin: 0; text-wrap: balance; letter-spacing: -.01em }
.meta { color: var(--muted); font-size: 13px; font-variant-numeric: tabular-nums }
.how { max-width: 66ch; margin: 0; color: var(--ink) }
.bar { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5; background: var(--paper);
  display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center; padding-block: 14px; border-bottom: 1px solid var(--line) }
.tally { display: flex; gap: 16px; font-size: 13px; color: var(--muted); font-variant-numeric: tabular-nums }
.tally b { color: var(--ink); font-weight: 600 }
.filters { display: flex; flex-wrap: wrap; gap: 6px; margin-left: auto }
.chip { font: 500 12px/1 var(--body); padding: 8px 12px; border-radius: 999px; border: 1px solid var(--line);
  background: transparent; color: var(--ink); cursor: pointer }
.chip[aria-pressed="true"] { background: var(--ink); color: var(--paper); border-color: var(--ink) }
.grid { display: grid; gap: 28px 22px; grid-template-columns: repeat(auto-fill, minmax(min(100%, 420px), 1fr)); padding-top: 26px }
.card { display: grid; gap: 10px; min-width: 0 }
.frame { position: relative; display: block; width: 100%; aspect-ratio: 3 / 2; background: var(--sheet); border: 0; padding: 0; cursor: zoom-in; overflow: hidden }
.frame img { width: 100%; height: 100%; object-fit: cover; display: block }
.frame .num { position: absolute; left: 10px; top: 10px; font: 600 11px/1 var(--body); letter-spacing: .08em;
  background: rgba(0,0,0,.55); color: #fff; padding: 5px 7px }
.frame[data-status="approved"] { outline: 3px solid var(--ok); outline-offset: -3px }
.frame[data-status="changes"] { outline: 3px solid var(--flag); outline-offset: -3px }
.row { display: flex; gap: 10px; align-items: baseline; justify-content: space-between; flex-wrap: wrap }
.name { font: 500 20px/1.1 var(--display) }
.exif { font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums }
.tags { display: flex; flex-wrap: wrap; gap: 6px }
.tag { font-size: 11px; letter-spacing: .03em; padding: 3px 7px; border: 1px solid var(--line); color: var(--muted) }
.actions { display: flex; gap: 8px; flex-wrap: wrap }
.btn { font: 500 13px/1 var(--body); padding: 9px 13px; border: 1px solid var(--line); background: var(--sheet); color: var(--ink); cursor: pointer }
.btn.ok[aria-pressed="true"] { background: var(--ok-bg); border-color: var(--ok); color: var(--ok) }
.btn.flag[aria-pressed="true"] { background: var(--flag-bg); border-color: var(--flag); color: var(--flag) }
.btn:disabled { opacity: .45; cursor: default }
textarea { width: 100%; min-height: 64px; resize: vertical; font: 14px/1.45 var(--body); color: var(--ink); background: var(--sheet);
  border: 1px solid var(--line); padding: 9px 10px }
.saved { font-size: 12px; color: var(--muted); min-height: 1em }
:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px }
.note-ro { font-size: 13px; color: var(--muted) }
/* compare */
.lb { position: fixed; inset: 0; z-index: 20; background: #0b0b0b; color: #eee; display: grid; grid-template-rows: auto 1fr auto;
  padding: calc(10px + env(safe-area-inset-top, 0px)) 16px calc(12px + env(safe-area-inset-bottom, 0px)) }
.lb header { border: 0; padding: 0 0 8px; display: flex; gap: 12px; align-items: center; justify-content: space-between }
.lb .t { font: 500 22px/1 var(--display) }
.lb .hint { font-size: 12px; color: #aaa }
.lb .ctl { display: flex; gap: 6px }
.lb button { font: 500 13px/1 var(--body); background: #222; color: #eee; border: 1px solid #333; padding: 9px 12px; cursor: pointer }
.stage { position: relative; min-height: 0; display: grid; place-items: center; user-select: none; touch-action: none }
.cmp { position: relative; max-width: 100%; max-height: 100%; aspect-ratio: var(--ar); width: min(100%, calc((100vh - 140px) * var(--arn))) }
.cmp img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; display: block }
.cmp .top { clip-path: inset(0 0 0 var(--x)) }
.cmp .handle { position: absolute; top: 0; bottom: 0; left: var(--x); width: 2px; background: #fff; transform: translateX(-1px) }
.cmp .handle::after { content: ""; position: absolute; top: 50%; left: 50%; width: 34px; height: 34px; border-radius: 50%;
  border: 2px solid #fff; background: rgba(0,0,0,.35); transform: translate(-50%, -50%) }
.cmp .lab { position: absolute; top: 10px; font: 600 11px/1 var(--body); letter-spacing: .12em; text-transform: uppercase;
  background: rgba(0,0,0,.55); padding: 6px 8px }
.cmp .lab.l { left: 10px } .cmp .lab.r { right: 10px }
input[type=range].slider { width: 100%; margin-top: 10px; accent-color: #fff }
@media (prefers-reduced-motion: no-preference) { .frame img { transition: transform .4s ease } .frame:hover img { transform: scale(1.015) } }
</style>

<div class="wrap">
  <header>
    <div class="brand">Keyframe Media · Signature Day</div>
    <h1>__HEADING__</h1>
    <div class="meta">__META__</div>
    <p class="how">Tap a frame to compare it with the camera's own JPEG of the middle exposure. Mark each image
      <b>Approve</b> or <b>Needs changes</b> and say what you'd like different. Claude reads these notes back and re-edits.</p>
  </header>
  <div class="bar" role="toolbar" aria-label="Review filters">
    <div class="tally" aria-live="polite"><span><b id="cA">0</b> approved</span><span><b id="cC">0</b> need changes</span><span><b id="cU">__COUNT__</b> to review</span></div>
    <div class="filters">
      <button class="chip" data-f="all" aria-pressed="true">All __COUNT__</button>
      <button class="chip" data-f="todo" aria-pressed="false">To review</button>
      <button class="chip" data-f="changes" aria-pressed="false">Needs changes</button>
      <button class="chip" data-f="approved" aria-pressed="false">Approved</button>
      <button class="chip" data-f="retouch" aria-pressed="false">Retouched __RETOUCHED__</button>
    </div>
  </div>
  <main class="grid" id="grid"></main>
</div>

<div class="lb" id="lb" hidden role="dialog" aria-modal="true" aria-label="Before and after">
  <header><div><div class="t" id="lbT"></div><div class="hint">Drag to compare · ← → next · Esc close</div></div>
    <div class="ctl"><button id="lbPrev" aria-label="Previous image">←</button><button id="lbNext" aria-label="Next image">→</button><button id="lbX">Close</button></div></header>
  <div class="stage" id="stage"><div class="cmp" id="cmp" style="--x:50%">
    <img id="imB" alt="Camera JPEG, middle exposure"><img id="imA" class="top" alt="Finished edit">
    <div class="handle"></div><span class="lab l">Camera</span><span class="lab r">Keyframe</span></div></div>
  <input id="sl" class="slider" type="range" min="0" max="100" value="50" aria-label="Compare position">
</div>

<script>
const ITEMS = __DATA__;
const state = {};   // key -> {status, note}
let filter = "all", db = null, canWrite = null, readOnly = false;
const grid = document.getElementById("grid");

function card(it, i) {
  const c = document.createElement("article"); c.className = "card"; c.dataset.key = it.key;
  const ex = it.exif || {};
  const exif = [ex.shutters ? ex.shutters + " s" : "", ex.f ? "f/" + ex.f : "", ex.iso ? "ISO " + ex.iso : "", ex.focal || ""].filter(Boolean).join(" · ");
  c.innerHTML = `
    <button class="frame" aria-label="Compare ${it.key}"><img loading="lazy" decoding="async" alt=""><span class="num"></span></button>
    <div class="row"><span class="name"></span><span class="exif"></span></div>
    <div class="tags"></div>
    <div class="actions"><button class="btn ok" aria-pressed="false">Approve</button><button class="btn flag" aria-pressed="false">Needs changes</button></div>
    <textarea id="note-${it.key}" placeholder="What should change? e.g. a touch brighter, warmer floor, remove the bin by the door" hidden></textarea>
    <div class="saved"></div>`;
  c.querySelector("img").src = it.after;
  c.querySelector(".num").textContent = it.n;
  c.querySelector(".name").textContent = it.key;
  c.querySelector(".exif").textContent = exif;
  const tags = c.querySelector(".tags");
  for (const n of it.notes) { const t = document.createElement("span"); t.className = "tag"; t.textContent = n; tags.append(t); }
  c.querySelector(".frame").onclick = () => openLB(i);
  c.querySelector(".ok").onclick = () => setStatus(it.key, state[it.key]?.status === "approved" ? null : "approved");
  c.querySelector(".flag").onclick = () => setStatus(it.key, state[it.key]?.status === "changes" ? null : "changes");
  const ta = c.querySelector("textarea");
  let timer; ta.oninput = () => { clearTimeout(timer); timer = setTimeout(() => saveNote(it.key, ta.value), 700); };
  return c;
}
ITEMS.forEach((it, i) => grid.append(card(it, i)));

function paint() {
  let a = 0, ch = 0;
  for (const it of ITEMS) {
    const s = state[it.key] || {}; const el = grid.querySelector(`[data-key="${it.key}"]`);
    if (s.status === "approved") a++; if (s.status === "changes") ch++;
    el.querySelector(".frame").dataset.status = s.status || "";
    el.querySelector(".ok").setAttribute("aria-pressed", s.status === "approved");
    el.querySelector(".flag").setAttribute("aria-pressed", s.status === "changes");
    const ta = el.querySelector("textarea");
    ta.hidden = !(s.status === "changes" || (s.note && s.note.length));
    if (document.activeElement !== ta) ta.value = s.note || "";
    ta.readOnly = readOnly;
    el.querySelectorAll(".btn").forEach(b => b.disabled = readOnly || !db);
    const show = filter === "all" || (filter === "todo" && !s.status) || (filter === s.status) ||
      (filter === "retouch" && it.notes.some(n => /removed/.test(n)));
    el.hidden = !show;
  }
  document.getElementById("cA").textContent = a; document.getElementById("cC").textContent = ch;
  document.getElementById("cU").textContent = ITEMS.length - a - ch;
}
document.querySelectorAll(".chip").forEach(b => b.onclick = () => {
  filter = b.dataset.f; document.querySelectorAll(".chip").forEach(x => x.setAttribute("aria-pressed", x === b)); paint(); });

async function write(key, patch) {
  if (!db || readOnly) return;
  const el = grid.querySelector(`[data-key="${key}"] .saved`);
  const next = { ...(state[key] || {}), ...patch, updatedAt: new Date().toISOString() };
  state[key] = next; paint(); el.textContent = "Saving…";
  try { await db.doc("reviews/" + key).set(next); el.textContent = "Saved"; }
  catch (e) {
    if (e && e.code === "invalid_argument") { readOnly = true; el.textContent = "You can view this review but not edit it."; paint(); }
    else el.textContent = "Couldn't save. Check your connection and try again.";
  }
}
function setStatus(key, status) { write(key, { status }); if (status === "changes") setTimeout(() => document.getElementById("note-" + key)?.focus(), 30); }
function saveNote(key, note) { if ((state[key]?.note || "") !== note) write(key, { note }); }

(async () => {
  try { db = await window.claude?.use?.("db"); } catch { db = null; }
  if (!db) { readOnly = true; paint(); return; }
  try { const u = await window.claude.use("user"); const c = u ? await u.can("data.write") : null; if (c === false) readOnly = true; } catch {}
  db.collection("reviews").onSnapshot(snap => {
    for (const d of snap.docs) state[d.id] = { ...d.data() };
    paint();
  }, () => {});
  paint();
})();
paint();

// before / after compare
const lb = document.getElementById("lb"), cmp = document.getElementById("cmp"), sl = document.getElementById("sl");
let cur = 0;
function openLB(i) {
  cur = (i + ITEMS.length) % ITEMS.length; const it = ITEMS[cur];
  document.getElementById("imA").src = it.after;
  document.getElementById("imB").src = it.before || it.after;
  cmp.style.setProperty("--ar", it.w + " / " + it.h); cmp.style.setProperty("--arn", it.w / it.h);
  document.getElementById("lbT").textContent = it.n + " · " + it.key;
  setX(50); lb.hidden = false; document.getElementById("lbX").focus();
}
function setX(v) { v = Math.max(0, Math.min(100, v)); cmp.style.setProperty("--x", v + "%"); sl.value = v; }
sl.oninput = () => setX(+sl.value);
const stage = document.getElementById("stage");
function drag(e) { const r = cmp.getBoundingClientRect(); setX((e.clientX - r.left) / r.width * 100); }
stage.addEventListener("pointerdown", e => { stage.setPointerCapture(e.pointerId); drag(e); });
stage.addEventListener("pointermove", e => { if (e.buttons) drag(e); });
document.getElementById("lbX").onclick = () => { lb.hidden = true; };
document.getElementById("lbPrev").onclick = () => openLB(cur - 1);
document.getElementById("lbNext").onclick = () => openLB(cur + 1);
document.addEventListener("keydown", e => { if (lb.hidden) return;
  if (e.key === "Escape") lb.hidden = true; if (e.key === "ArrowRight") openLB(cur + 1); if (e.key === "ArrowLeft") openLB(cur - 1); });
</script>
"""

if __name__ == "__main__":
    main()
