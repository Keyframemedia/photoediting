"""The Keyframe editing farm on Modal (modal.com).

The portal (Deliveries -> Photo edits, photo-edit-start) posts {job_id, token,
keys} to `start`. `start` asks the portal to confirm the token (nobody else can
start work), then spawns `run_job`, which plans the shoot and edits every bracket
at once, one `edit_bracket` machine each (8 cores, 16 GB), so a shoot takes about
as long as its slowest photo. Nothing runs, and nothing is billed, between jobs.

Deploy (also done by .github/workflows/deploy-farm.yml on every push):
    pip install modal && modal deploy worker/modal_app.py
The start address printed by the deploy goes in Photo edits -> Settings.
"""
from __future__ import annotations

import pathlib
import time

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent
APP = "keyframe-editing-farm"
app = modal.App(APP)


def _prefetch_skies():
    """Build the 360-degree sky domes into the image, so no job ever downloads one."""
    import sys
    sys.path.insert(0, "/root/skybuild")
    import skydome  # standalone copy: only skydome.py changing rebuilds this layer
    for names in skydome.LIBRARY.values():
        for name in names:
            skydome.load(name)
    for f in pathlib.Path(skydome.CACHE).glob("*.hdr"):
        f.unlink()  # the bands are all the pipeline reads


image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
    .env({"DEBIAN_FRONTEND": "noninteractive", "WINEDEBUG": "-all", "WINEPREFIX": "/root/.wine",
          "PYTHONPATH": "/root/kf", "KF_SKY_CACHE": "/root/.cache/keyframe_skies", "HOME": "/root"})
    .apt_install("libimage-exiftool-perl", "xvfb", "xauth", "unzip", "curl", "ca-certificates",
                 "libgl1", "libglib2.0-0")
    .run_commands("dpkg --add-architecture i386 && apt-get update -qq && "
                  "apt-get install -y -qq --no-install-recommends wine wine64 wine32:i386")
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("numpy>=2.0", "opencv-python-headless>=4.10", "rawpy>=0.22", "scipy", "numexpr",
                 "pillow", "transformers", "onnxruntime", "fastapi[standard]")
    # models (LaMa, UperNet, Mask R-CNN) and the Adobe DNG Converter under Wine
    .add_local_file(REPO / "scripts" / "setup_worker.sh", "/root/setup_worker.sh", copy=True)
    .run_commands("bash /root/setup_worker.sh")
    .add_local_file(REPO / "keyframe_hdr" / "skydome.py", "/root/skybuild/skydome.py", copy=True)
    .run_function(_prefetch_skies)
    # code last: a code change redeploys in seconds without rebuilding the layers above
    .add_local_dir(REPO / "keyframe_hdr", "/root/kf/keyframe_hdr", ignore=["__pycache__", "luts/*.part*"])
    .add_local_dir(REPO / "worker", "/root/kf/worker", ignore=["__pycache__"])
)

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]")
    .env({"PYTHONPATH": "/root/kf"})
    .add_local_dir(REPO / "worker", "/root/kf/worker", ignore=["__pycache__"])
)


@app.function(image=image, cpu=8.0, memory=16384, timeout=45 * 60, retries=1, max_containers=60)
def edit_bracket(spec: dict) -> dict:
    from worker import farm
    return farm.edit_bracket(spec)


def _fan_out(fn, specs):
    """Run every bracket at once; yield results as they finish. Closing the
    generator (Stop pressed) cancels whatever is still running."""
    calls = {i: edit_bracket.spawn(s) for i, s in enumerate(specs)}
    try:
        while calls:
            for i, c in list(calls.items()):
                try:
                    res = c.get(timeout=0)
                except TimeoutError:
                    continue
                except Exception as e:  # the bracket's machine failed twice
                    res = e
                del calls[i]
                yield res
            if calls:
                time.sleep(3)
    finally:
        for c in calls.values():
            try:
                c.cancel()
            except Exception:
                pass


@app.function(image=image, cpu=2.0, memory=4096, timeout=4 * 3600)
def run_job(job_id: str, token: str, keys: list[str] | None = None) -> dict:
    from worker import farm
    return farm.run(job_id, token, keys, _fan_out, parallel=True)


@app.function(image=web_image, timeout=60)
@modal.fastapi_endpoint(method="POST", label="keyframe-farm-start")
def start(body: dict):
    from fastapi import HTTPException

    from worker.farm import Portal, Refused
    job_id, token, keys = body.get("job_id"), body.get("token"), body.get("keys") or None
    if not isinstance(job_id, str) or not isinstance(token, str):
        raise HTTPException(400, "job_id and token are required")
    if keys is not None and (not isinstance(keys, list) or not all(isinstance(k, str) for k in keys)):
        raise HTTPException(400, "keys must be a list of photo names")
    try:
        Portal(job_id, token).call("get")  # the portal confirms this run's token
    except Refused:
        raise HTTPException(403, "The portal didn't confirm this job's token")
    call = run_job.spawn(job_id, token, keys)
    return {"call_id": call.object_id}
