"""What is in the finished image, for the automatic edits that need to know.

One ADE20K segmentation of the finished (display) image, at preview size, feeds:
  * reflections: people found by the detector are removed automatically when
    they sit on glass, a mirror or a screen (the photographer's reflection);
    uncertain ones, and people standing in the open, are flagged for review;
    people in artwork are left alone;
  * TV screens shown black;
  * lawn greening.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import sky as skymod

# ADE20K class ids (as used by sky.py)
WINDOWPANE, GRASS, DOOR, PAINTING, MIRROR, FIELD = 8, 9, 14, 22, 27, 29
TV, POSTER, SCREEN, CRT, MONITOR, GLASS = 89, 100, 130, 141, 143, 147
BOOK, PILLOW, SCULPTURE, BULLETIN = 67, 57, 132, 144
REFLECTIVE = (WINDOWPANE, DOOR, MIRROR, GLASS)
SCREENS = (TV, SCREEN, CRT, MONITOR)
ART = (PAINTING, POSTER, BOOK, PILLOW, SCULPTURE, BULLETIN)  # prints, covers, cushions

# reflection rules (detector score, share of the person on a reflective surface)
AUTO_SCORE, AUTO_REFLECT = 0.5, 0.35
REVIEW_SCORE = 0.3


def segment(v: np.ndarray, size: int = 640) -> np.ndarray:
    """ADE20K class probabilities (C, h, w) of a finished sRGB float image."""
    H, W = v.shape[:2]
    s = size / min(H, W)
    img = cv2.resize(v, (max(1, int(round(W * s))), max(1, int(round(H * s)))), interpolation=cv2.INTER_AREA)
    return skymod._segment((np.clip(img, 0, 1) * 255).astype(np.uint8))


def _group(probs: np.ndarray, ids) -> np.ndarray:
    return np.clip(sum(probs[i] for i in ids), 0, 1)


def _box_share(prob: np.ndarray, box) -> float:
    h, w = prob.shape
    x0, y0, x1, y1 = box
    sub = prob[int(y0 * h):max(int(y0 * h) + 1, int(np.ceil(y1 * h))),
               int(x0 * w):max(int(x0 * w) + 1, int(np.ceil(x1 * w)))]
    return float(sub.mean()) if sub.size else 0.0


def _overlaps(a, b, min_iou: float = 0.1) -> bool:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return False
    area = lambda r: max(1e-9, (r[2] - r[0]) * (r[3] - r[1]))
    return inter / min(area(a), area(b)) >= min_iou


def _where(v: np.ndarray, probs: np.ndarray, box) -> tuple[str, dict]:
    """What a detected person is on: 'glass', 'screen', 'artwork' or 'room'.

    Small things (a TV across the room, a mirror, a book cover) are lost in the
    whole-image segmentation, so the area around the person is segmented again,
    zoomed in, and its classes read inside the box."""
    H, W = v.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    cx0, cy0 = max(0.0, x0 - 1.5 * bw), max(0.0, y0 - 1.0 * bh)
    cx1, cy1 = min(1.0, x1 + 1.5 * bw), min(1.0, y1 + 1.0 * bh)
    crop = v[int(cy0 * H):max(int(cy0 * H) + 16, int(cy1 * H)), int(cx0 * W):max(int(cx0 * W) + 16, int(cx1 * W))]
    cp = segment(crop, size=384) if min(crop.shape[:2]) >= 16 else probs
    if cp is probs:
        inner = box
    else:  # the box in crop coordinates
        sw, sh = max(1e-6, cx1 - cx0), max(1e-6, cy1 - cy0)
        inner = [(x0 - cx0) / sw, (y0 - cy0) / sh, (x1 - cx0) / sw, (y1 - cy0) / sh]
    share = {"glass": max(_box_share(_group(cp, REFLECTIVE), inner), _box_share(_group(probs, REFLECTIVE), box)),
             "screen": _box_share(_group(cp, SCREENS), inner),
             "art": _box_share(_group(cp, ART), inner)}
    if share["art"] > 0.35:
        where = "artwork"
    elif share["screen"] > 0.3:
        where = "screen"
    elif share["glass"] > AUTO_REFLECT:
        where = "glass"
    else:
        where = "room"
    return where, share


def classify_people(v: np.ndarray, probs: np.ndarray, keep: list | None = None,
                    min_score: float = 0.25, aerial: bool = False) -> list[dict]:
    """People candidates with what to do about each:
    action 'remove' (a confident person on glass, a mirror or a dark screen: the
    photographer's reflection), 'review' (to check: uncertain, on a lit screen, or
    standing in the room) or 'ignore' (in artwork, or kept by a reviewer). Boxes
    are normalised to the image. Small figures in the open, and any in the open on
    a drone shot, are passers-by in the street: not reported."""
    from .retouch import detect_people
    keep = keep or []
    Y = v @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    H, W = Y.shape
    out = []
    for d in detect_people(v, min_score=min_score):
        box = [round(float(x), 4) for x in d["box"]]
        sc = float(d["score"])
        if aerial:
            where, share = "room", {}
        else:
            where, share = _where(v, probs, box)
        # how dark the object around the person is: a switched-off TV is near
        # black, a painting, print or book cover of a person is lit and coloured
        x0, y0, x1, y1 = box
        pad = 0.5
        ring = Y[int(max(0, y0 - pad * (y1 - y0)) * H):int(min(1, y1 + pad * (y1 - y0)) * H),
                 int(max(0, x0 - pad * (x1 - x0)) * W):int(min(1, x1 + pad * (x1 - x0)) * W)]
        dark = bool(ring.size) and float(np.median(ring)) < 0.18
        if where in ("artwork", "screen") and dark:
            where = "screen"  # a figure on a dark screen is a reflection in it
        if any(_overlaps(box, k) for k in keep) or where == "artwork":
            action = "ignore"
        elif where in ("glass", "screen") and sc >= AUTO_SCORE and (where == "glass" or dark):
            action = "remove"
        elif where == "screen":
            action = "review"  # a lit screen: a picture, or a reflection to check
        elif where == "room" and (aerial or box[3] - box[1] < 0.06 or sc < REVIEW_SCORE):
            continue  # a passer-by, or a faint false detection
        else:
            action = "review"
        out.append({"box": box, "score": round(sc, 3), "where": where, "action": action,
                    **{k: round(v_, 2) for k, v_ in share.items()}})
    return out


def screen_mask(probs: np.ndarray, shape: tuple[int, int]) -> list[np.ndarray]:
    """Quadrilaterals (full-size pixel corners) of TV screens: large, rectangular
    regions the model calls a screen."""
    m = (_group(probs, SCREENS) > 0.5).astype(np.uint8)
    h, w = m.shape
    H, W = shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    quads = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 0.004 * h * w:
            continue
        comp = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        if rw * rh <= 0 or cv2.contourArea(c) / (rw * rh) < 0.8:
            continue  # not a clean rectangle: a reflection, a cabinet, a window
        ar = max(rw, rh) / max(1e-6, min(rw, rh))
        if not 1.2 <= ar <= 2.4:
            continue  # screens are 16:9-ish
        box = cv2.boxPoints(rect) * np.array([W / w, H / h], np.float32)
        quads.append(box)
    return quads


def black_screens(v: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, int]:
    """Show TVs switched off: glossy black with a faint diagonal sheen. A screen
    that is already dark is left as it is."""
    H, W = v.shape[:2]
    n = 0
    for q in screen_mask(probs, (H, W)):
        m = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(m, q.astype(np.int32), 1)
        # shrink a touch so the bezel stays, feather a pixel or two
        k = max(3, int(0.004 * max(H, W)))
        m = cv2.erode(m, np.ones((k, k), np.uint8))
        if m.sum() < 100:
            continue
        Y = v @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        if float(np.median(Y[m > 0])) < 0.12:
            continue  # already off
        x0, y0, bw, bh = cv2.boundingRect(m)
        yy, xx = np.mgrid[0:bh, 0:bw].astype(np.float32)
        sheen = 0.035 + 0.03 * np.clip(1 - ((xx / max(bw, 1)) + (yy / max(bh, 1))), 0, 1) ** 2
        fill = np.zeros((H, W, 3), np.float32)
        fill[y0:y0 + bh, x0:x0 + bw] = sheen[..., None] * np.array([0.95, 0.98, 1.0], np.float32)
        a = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.2)[..., None]
        v = v * (1 - a) + fill * a
        n += 1
    return v, n


def green_lawn(v: np.ndarray, probs: np.ndarray, strength: float = 0.6) -> tuple[np.ndarray, float]:
    """Pull tired lawn (straw, olive) toward a healthy green, keeping its texture
    and brightness. Only pixels the model calls grass, and only yellow-green hues,
    so paths, timber and garden beds are untouched."""
    from .grade import gamut_map_srgb, oklab_to_rec2020, rec2020_to_oklab
    from .tonemap import srgb_decode, srgb_encode
    H, W = v.shape[:2]
    g = cv2.resize(_group(probs, (GRASS, FIELD)), (W, H), interpolation=cv2.INTER_LINEAR)
    g = np.clip((g - 0.4) / 0.3, 0, 1)
    share = float(g.mean())
    if share < 0.005:
        return v, 0.0
    M = np.array([[0.6274040, 0.3292820, 0.0433136], [0.0690970, 0.9195400, 0.0113612],
                  [0.0163916, 0.0880132, 0.8955950]], np.float32)
    out = v.copy()
    for y0 in range(0, H, 512):
        sl = slice(y0, y0 + 512)
        w = g[sl]
        if w.max() <= 0:
            continue
        lab = rec2020_to_oklab(srgb_decode(v[sl]) @ M.T)
        a, b = lab[..., 1], lab[..., 2]
        h = np.degrees(np.arctan2(b, a))
        C = np.hypot(a, b)
        # lawn hues: straw (~85 deg) to green (~140 deg); target a fresh green
        band = np.clip(1 - np.abs(h - 112) / 45, 0, 1) * np.clip(C / 0.02, 0, 1)
        t = (w * band * strength)
        h2 = h + (138 - h) * t
        C2 = C * (1 + 0.35 * t)
        lab[..., 1] = C2 * np.cos(np.radians(h2))
        lab[..., 2] = C2 * np.sin(np.radians(h2))
        out[sl] = srgb_encode(np.clip(gamut_map_srgb(oklab_to_rec2020(lab).astype(np.float32)), 0, 1))
    return out, round(share, 4)
