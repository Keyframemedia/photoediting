"""House-look 3D LUTs, learned from Keyframe's own delivered images.

A LUT maps the pipeline's finished sRGB colour to the colour Keyframe delivers,
fitted on matched pairs (the same bracket, edited by the pipeline and by hand).
The tone mapping, windows and local contrast stay the pipeline's; the LUT carries
the colour rendering: how white walls, timber, greens and sky blue are drawn.

Fitting is a smoothed least-squares problem on a coarse lattice (13^3) with
trilinear weights, so cells the samples never reach are filled smoothly from
their neighbours instead of being left at identity. The result is then baked
through a cubic spline into a dense 65^3 table: a trilinear LUT has a slope
kink at every cell boundary, and on a smooth ceiling or sky a fine, freely
fitted lattice shows those kinks (and any wiggle between nodes) as contour
rings. Coarse nodes + spline + dense table keeps every gradient smooth.
"""
from __future__ import annotations

import os

import numpy as np

LUT_DIR = os.path.join(os.path.dirname(__file__), "luts")


def _corners(rgb: np.ndarray, n: int):
    x = np.clip(rgb, 0, 1) * (n - 1)
    i0 = np.minimum(np.floor(x).astype(np.int64), n - 2)
    f = (x - i0).astype(np.float32)
    return i0, f


def apply(rgb: np.ndarray, lut: np.ndarray, strength: float = 1.0, rows: int = 512) -> np.ndarray:
    """Trilinear lookup of an (n, n, n, 3) LUT on sRGB values in [0, 1] (in place, by strips)."""
    n = lut.shape[0]
    flat = lut.reshape(-1, 3).astype(np.float32)
    H = rgb.shape[0]
    for r0 in range(0, H, rows):
        blk = rgb[r0:r0 + rows]
        sh = blk.shape
        p = blk.reshape(-1, 3)
        i0, f = _corners(p, n)
        out = np.zeros_like(p, dtype=np.float32)
        for dr in (0, 1):
            wr = f[:, 0] if dr else 1 - f[:, 0]
            for dg in (0, 1):
                wg = f[:, 1] if dg else 1 - f[:, 1]
                for db in (0, 1):
                    wb = f[:, 2] if db else 1 - f[:, 2]
                    idx = ((i0[:, 0] + dr) * n + (i0[:, 1] + dg)) * n + (i0[:, 2] + db)
                    out += flat[idx] * (wr * wg * wb)[:, None]
        if strength != 1.0:
            out = p + (out - p) * np.float32(strength)
        rgb[r0:r0 + rows] = np.clip(out, 0, 1).reshape(sh)
    return rgb


def fit(src: np.ndarray, dst: np.ndarray, n: int = 13, smooth: float = 1.0, weight: np.ndarray | None = None) -> np.ndarray:
    """Least-squares LUT with a Laplacian smoothness prior toward identity-shaped
    differences: minimise sum w*|T(src) - dst|^2 + smooth * N/n^3 * |L (T - I)|^2."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    src = src.reshape(-1, 3).astype(np.float32)
    dst = dst.reshape(-1, 3).astype(np.float32)
    w = np.ones(len(src), np.float32) if weight is None else weight.reshape(-1).astype(np.float32)
    N, M = len(src), n ** 3
    i0, f = _corners(src, n)
    rows, cols, vals = [], [], []
    for dr in (0, 1):
        wr = f[:, 0] if dr else 1 - f[:, 0]
        for dg in (0, 1):
            wg = f[:, 1] if dg else 1 - f[:, 1]
            for db in (0, 1):
                wb = f[:, 2] if db else 1 - f[:, 2]
                rows.append(np.arange(N)); cols.append(((i0[:, 0] + dr) * n + (i0[:, 1] + dg)) * n + (i0[:, 2] + db))
                vals.append(wr * wg * wb)
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, M))
    W = sp.diags(w)
    # 3-D grid Laplacian
    g = np.arange(M).reshape(n, n, n)
    lr, lc, lv = [], [], []
    for ax in range(3):
        a = np.moveaxis(g, ax, 0)
        u, v = a[:-1].ravel(), a[1:].ravel()
        lr += [u, v, u, v]; lc += [u, v, v, u]
        lv += [np.ones_like(u), np.ones_like(v), -np.ones_like(u), -np.ones_like(v)]
    L = sp.csr_matrix((np.concatenate(lv).astype(np.float32), (np.concatenate(lr), np.concatenate(lc))), shape=(M, M))
    ident = np.stack(np.meshgrid(*[np.linspace(0, 1, n)] * 3, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
    lam = smooth * w.sum() / M
    lhs = (A.T @ W @ A + lam * (L.T @ L)).tocsc()
    out = np.empty((M, 3), np.float32)
    solve = spla.factorized(lhs)
    for c in range(3):
        rhs = A.T @ (w * dst[:, c]) + lam * (L.T @ (L @ ident[:, c]))
        out[:, c] = solve(rhs)
    return np.clip(out, 0, 1).reshape(n, n, n, 3)


def bake(lut: np.ndarray, m: int = 65) -> np.ndarray:
    """Resample a fitted lattice to an m^3 table through a cubic spline."""
    from scipy.ndimage import map_coordinates
    n = lut.shape[0]
    g = np.linspace(0, n - 1, m)
    grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), 0)
    out = np.stack([map_coordinates(lut[..., c].astype(np.float64), grid, order=3, mode="nearest")
                    for c in range(3)], -1)
    return np.clip(out, 0, 1).astype(np.float32)


def load(name: str) -> np.ndarray | None:
    p = os.path.join(LUT_DIR, name if name.endswith(".npy") else name + ".npy")
    return np.load(p).astype(np.float32) if os.path.exists(p) else None
