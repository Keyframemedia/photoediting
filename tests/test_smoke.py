"""Fast smoke tests on synthetic data (no RAWs or models needed).

    python -m pytest tests/ -q
"""
import numpy as np

from keyframe_hdr import geometry, merge, presets, render
from keyframe_hdr.raw import Frame


def _scene(h=360, w=540, seed=0):
    """A synthetic interior: dim room, a bright 'window' 6 stops hotter, some texture."""
    rng = np.random.default_rng(seed)
    y = np.full((h, w), 0.02, np.float32)
    y[60:200, 330:500] = 1.3  # window
    y += rng.normal(0, 0.002, y.shape).astype(np.float32)
    rgb = np.stack([y * 1.05, y, y * 0.9], axis=-1)
    return np.maximum(rgb, 0)


def test_merge_recovers_radiance():
    scene = _scene()
    frames = []
    for ev in (-3, 0, 3):
        e = 2.0 ** ev
        img = scene * e * 0.1
        clip = (img.max(axis=2) > 0.95).astype(np.float32)
        frames.append(Frame("x", np.clip(img, 0, 1).astype(np.float32), clip, e, {}))
    hdr, info = merge.merge(frames)
    ref = scene * 0.1  # units of the middle exposure
    win = (slice(80, 180), slice(350, 480))
    room = (slice(250, 340), slice(20, 300))
    assert abs(hdr[win].mean() / ref[win].mean() - 1) < 0.05   # window recovered from short frame
    assert abs(hdr[room].mean() / ref[room].mean() - 1) < 0.05  # room from long frame


def test_render_day_and_night():
    hdr = _scene()
    for p in (presets.DAY, presets.NIGHT):
        v = render.render(hdr, p)
        assert v.shape == hdr.shape and v.dtype == np.float32
        assert np.isfinite(v).all() and v.min() >= 0 and v.max() <= 1
        assert v[80:180, 350:480].mean() > v[250:340, 20:300].mean()  # window stays brighter


def test_render_inplace_matches():
    hdr = _scene()
    a = render.render(hdr, presets.DAY)
    b = render.render(hdr.copy(), presets.DAY, inplace=True)
    assert np.abs(a - b).max() < 1e-5


def test_bracket_grouping():
    metas = []
    for b in range(4):
        for i, t in enumerate((1 / 250, 1 / 2000, 1 / 30)):
            metas.append({"FileName": f"F{b}{i}", "DateTimeOriginal": f"2026:09:01 14:4{b}:00",
                          "SubSecTimeOriginal": f"{10 + i * 5}", "ExposureTime": t, "FNumber": 8,
                          "ISO": 320, "FocalLength": 14, "Model": "Z8"})
    groups = merge.group_brackets(metas)
    assert [len(g) for g in groups] == [3, 3, 3, 3]


def test_upright_rotation_is_identity_for_vertical():
    R = geometry._rotation_to_y(np.array([0.0, 1.0, 0.0]))
    assert np.allclose(R, np.eye(3))
