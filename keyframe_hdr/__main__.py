"""Command line: python -m keyframe_hdr <input_dir> <output_dir> [--style day|twilight]"""
import argparse

from .pipeline import run
from .presets import job_overrides


def main():
    ap = argparse.ArgumentParser(description="Keyframe HDR: bracketed RAWs -> finished real estate images")
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--style", default="day", choices=["day", "twilight", "night"],
                    help="night is the same look as twilight")
    ap.add_argument("--sky", default="original", choices=["original", "clouds", "clear"],
                    help="replace the sky: clouds / clear blue (day) or clear / clouds dusk (twilight)")
    ap.add_argument("--look", default="natural", choices=["natural", "purple"], help="twilight look")
    ap.add_argument("--no-lights", action="store_true", help="twilight: don't enhance the lights")
    ap.add_argument("--seed", default="0", help="per-shoot seed (picks the sky dome and its rotation)")
    ap.add_argument("--max-mb", type=float, default=None, help="size cap for full-res JPEGs (e.g. 10)")
    ap.add_argument("--only", nargs="*", help="process only brackets containing these file stems")
    ap.add_argument("--half", action="store_true", help="half-resolution preview run (fast)")
    ap.add_argument("--web-size", type=int, default=2560, help="long edge of web exports")
    ap.add_argument("--work-dir", help="where converted DNGs are cached")
    ap.add_argument("--jobs", type=int, default=1, help="brackets processed in parallel (~8 GB RAM each)")
    ap.add_argument("--force", action="store_true", help="re-process brackets that already have output")
    ap.add_argument("--reverse", action="store_true", help="work through brackets last-to-first "
                    "(run a second process with this to share one output folder)")
    a = ap.parse_args()
    style = "day" if a.style == "day" else "twilight"
    opts = job_overrides(style, a.sky, a.look, not a.no_lights)
    opts["sky_seed"] = a.seed
    run(a.input_dir, a.output_dir, style, half=a.half, only=a.only,
        web_long_edge=a.web_size, work_dir=a.work_dir, jobs=a.jobs, resume=not a.force, reverse=a.reverse,
        options=opts, max_mb=a.max_mb)


if __name__ == "__main__":
    main()
