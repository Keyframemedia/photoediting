"""Retouching: remove photographers/people (and their gear) from reflections.

Detection: torchvision Mask R-CNN (COCO 'person', BSD licence) with a low score
threshold, because reflected people in glass are faint.
Removal: LaMa inpainting (Apache-2.0, ONNX) on a context crop around each
region, composited back with a feathered mask and matched film grain.

Edits are given per image in edits.json, coordinates normalised to the output:
  {"retouch": [{"box": [x0, y0, x1, y1], "mode": "person"},   # find people in box, remove
               {"box": [x0, y0, x1, y1], "mode": "fill"},     # remove the whole box
               {"box": [x0, y0, x1, y1], "mode": "neutral"},  # take colour out (moire brush)
               {"poly": [[x, y], ...]}]}                        # remove a polygon
"""
from __future__ import annotations

import os

import cv2
import numpy as np

LAMA_PATH = os.environ.get("KF_LAMA", os.path.expanduser("~/.cache/keyframe_models/lama_fp32.onnx"))
_M = {}


def _lama():
    if "lama" not in _M:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, os.cpu_count() or 1)
        _M["lama"] = ort.InferenceSession(LAMA_PATH, so, providers=["CPUExecutionProvider"])
    return _M["lama"]


def _maskrcnn():
    if "rcnn" not in _M:
        import torch
        import torchvision
        w = torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        m = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=w, box_score_thresh=0.15).eval()
        _M["rcnn"] = (m, torch)
    return _M["rcnn"]


def detect_people(v: np.ndarray, min_score: float = 0.25, long_edge: int = 1600) -> list[dict]:
    """People anywhere in the image (normalised boxes, scores, soft masks)."""
    m, torch = _maskrcnn()
    H, W = v.shape[:2]
    s = min(1.0, long_edge / max(H, W))
    small = cv2.resize(v, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else v
    t = torch.from_numpy(np.ascontiguousarray(small.transpose(2, 0, 1))).float()
    with torch.no_grad():
        out = m([t])[0]
    res = []
    h, w = small.shape[:2]
    for box, lab, sc, mk in zip(out["boxes"], out["labels"], out["scores"], out["masks"]):
        if int(lab) != 1 or float(sc) < min_score:
            continue
        x0, y0, x1, y1 = box.tolist()
        res.append({"score": round(float(sc), 3), "box": [x0 / w, y0 / h, x1 / w, y1 / h],
                    "mask": mk[0].numpy()})
    return res


def _person_mask_in_box(v: np.ndarray, box, grow: float = 0.012) -> np.ndarray:
    """Segment people inside a (normalised) box; returns a full-size bool mask.

    Runs the detector at two scales - the whole frame and a zoomed crop - and
    unions every person mask overlapping the box: faint reflections are often
    found whole at one scale and only in pieces at the other."""
    H, W = v.shape[:2]
    x0, y0, x1, y1 = box
    inner = np.zeros((H, W), bool)
    inner[int(y0 * H):int(np.ceil(y1 * H)), int(x0 * W):int(np.ceil(x1 * W))] = True
    full = np.zeros((H, W), np.float32)

    def add(dets, ox, oy, cw, ch):
        for d in dets:
            bx0, by0, bx1, by1 = d["box"]
            # overlap test in full-image normalised coords
            fx0, fy0 = (ox + bx0 * cw) / W, (oy + by0 * ch) / H
            fx1, fy1 = (ox + bx1 * cw) / W, (oy + by1 * ch) / H
            if fx1 < x0 or fx0 > x1 or fy1 < y0 or fy0 > y1:
                continue
            mk = cv2.resize(d["mask"], (cw, ch), interpolation=cv2.INTER_LINEAR)
            full[oy:oy + ch, ox:ox + cw] = np.maximum(full[oy:oy + ch, ox:ox + cw], mk)

    add(detect_people(v, min_score=0.15, long_edge=2000), 0, 0, W, H)
    pad = 0.25
    bx0, by0 = max(0, int((x0 - pad * (x1 - x0)) * W)), max(0, int((y0 - pad * (y1 - y0)) * H))
    bx1, by1 = min(W, int((x1 + pad * (x1 - x0)) * W)), min(H, int((y1 + pad * (y1 - y0)) * H))
    crop = v[by0:by1, bx0:bx1]
    add(detect_people(crop, min_score=0.15, long_edge=1024), bx0, by0, bx1 - bx0, by1 - by0)
    mask = (full > 0.2) & inner
    if mask.sum() < 50:  # nothing found: fall back to removing the whole box
        mask = inner
    r = max(3, int(grow * max(H, W)))
    return cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))) > 0


def inpaint(v: np.ndarray, mask: np.ndarray, context: float = 2.2) -> np.ndarray:
    """LaMa inpainting of `mask` (bool, full size) in sRGB float image `v`."""
    out = v.copy()
    H, W = v.shape[:2]
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    sess = _lama()
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 4:
            continue
        side = int(max(bw, bh) * context)
        side = max(side, 256)
        side = min(side, max(H, W))
        cx, cy = x + bw / 2, y + bh / 2
        x0 = int(np.clip(cx - side / 2, 0, max(0, W - side)))
        y0 = int(np.clip(cy - side / 2, 0, max(0, H - side)))
        x1, y1 = min(W, x0 + side), min(H, y0 + side)
        crop = out[y0:y1, x0:x1]
        cm = (lab[y0:y1, x0:x1] == i).astype(np.float32)
        img512 = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)
        m512 = (cv2.resize(cm, (512, 512), interpolation=cv2.INTER_LINEAR) > 0.01).astype(np.float32)
        res = sess.run(None, {"image": img512.transpose(2, 0, 1)[None].astype(np.float32),
                              "mask": m512[None, None]})[0][0].transpose(1, 2, 0)
        if res.max() > 2:
            res = res / 255.0
        res = np.clip(res, 0, 1).astype(np.float32)
        fill = cv2.resize(res, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
        # put back grain: the fill is upscaled from 512px and would look plastic
        ring = (cv2.dilate(cm, np.ones((15, 15), np.uint8)) > 0) & (cm == 0)
        if ring.sum() > 100:
            Yc = crop @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
            hp = Yc - cv2.GaussianBlur(Yc, (0, 0), 1.2)
            sd = float(np.std(hp[ring]))
            if sd > 1e-4:
                g = np.random.default_rng(i).normal(0, sd, fill.shape[:2]).astype(np.float32)
                g = g - cv2.GaussianBlur(g, (0, 0), 1.2)
                lum = np.clip(fill @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), 0, 1)
                fill = fill + (g * 0.8 * np.clip(lum / 0.35, 0.15, 1.0))[..., None]
        feather = max(1.0, 0.004 * side)
        a = cv2.GaussianBlur(cv2.dilate(cm, np.ones((3, 3), np.uint8)), (0, 0), feather)
        a = np.clip(a * 1.5, 0, 1)[..., None]
        out[y0:y1, x0:x1] = crop * (1 - a) + np.clip(fill, 0, 1) * a
    return out


def apply_edits(v: np.ndarray, edits: list[dict]) -> tuple[np.ndarray, list]:
    H, W = v.shape[:2]
    mask = np.zeros((H, W), bool)
    log = []
    for e in edits:
        if e.get("mode") == "neutral":
            # moire brush: remove false colour from fine grilles/fabrics, keep detail
            x0, y0, x1, y1 = e["box"]
            m = np.zeros((H, W), np.float32)
            m[int(y0 * H):int(np.ceil(y1 * H)), int(x0 * W):int(np.ceil(x1 * W))] = 1
            m = cv2.GaussianBlur(m, (0, 0), max(1.0, 0.002 * max(H, W)))[..., None]
            Y = (v @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))[..., None]
            v = v * (1 - m) + Y * m
            log.append({"box": e["box"], "mode": "neutral"})
            continue
        if "poly" in e:
            pts = (np.array(e["poly"], dtype=np.float32) * [W, H]).astype(np.int32)
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [pts], 1)
            r = max(3, int(0.004 * max(H, W)))
            mask |= cv2.dilate(m, np.ones((r, r), np.uint8)) > 0
            log.append({"poly": len(e["poly"])})
        elif e.get("mode", "person") == "person":
            m = _person_mask_in_box(v, e["box"], e.get("grow", 0.012))
            mask |= m
            log.append({"box": e["box"], "mode": "person", "pixels": int(m.sum())})
        else:
            x0, y0, x1, y1 = e["box"]
            mask[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = True
            log.append({"box": e["box"], "mode": "fill"})
    if mask.any():
        v = inpaint(v, mask)
    return v, log
