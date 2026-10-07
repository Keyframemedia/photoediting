"""Command line: python -m keyframe_hdr <input_dir> <output_dir> [--style day|night]"""
import argparse

from .pipeline import run


def main():
    ap = argparse.ArgumentParser(description="Keyframe HDR: bracketed RAWs -> finished real estate images")
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--style", default="day", choices=["day", "night"])
    ap.add_argument("--only", nargs="*", help="process only brackets containing these file stems")
    ap.add_argument("--half", action="store_true", help="half-resolution preview run (fast)")
    ap.add_argument("--web-size", type=int, default=2560, help="long edge of web exports")
    ap.add_argument("--work-dir", help="where converted DNGs are cached")
    ap.add_argument("--jobs", type=int, default=1, help="brackets processed in parallel (~8 GB RAM each)")
    ap.add_argument("--force", action="store_true", help="re-process brackets that already have output")
    ap.add_argument("--reverse", action="store_true", help="work through brackets last-to-first "
                    "(run a second process with this to share one output folder)")
    a = ap.parse_args()
    run(a.input_dir, a.output_dir, a.style, half=a.half, only=a.only,
        web_long_edge=a.web_size, work_dir=a.work_dir, jobs=a.jobs, resume=not a.force, reverse=a.reverse)


if __name__ == "__main__":
    main()
