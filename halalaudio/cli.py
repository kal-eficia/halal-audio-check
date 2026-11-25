"""halal-audio-check: flag music in a soundtrack.

  halal-audio-check final.mp4
  halal-audio-check *.mp4 --json
  halal-audio-check reel.mov --timeline reel-audio.png

Exit code 0 when every file is clean, 1 when music was found, 2 on errors.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .analyse import check_file, timestamp

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


def timeline_png(report, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (install matplotlib for --timeline)", file=sys.stderr)
        return
    t = [(w.start + w.end) / 2 for w in report.windows]
    fig, ax = plt.subplots(figsize=(10, 2.2), dpi=150)
    ax.fill_between(t, [w.score for w in report.windows], color="#0f8b8d", alpha=0.25, step="mid")
    ax.plot(t, [w.score for w in report.windows], color="#0f8b8d", lw=1.4, drawstyle="steps-mid")
    for seg in report.segments:
        ax.axvspan(seg.start, seg.end, color="#e0614a", alpha=0.18, lw=0)
    ax.axhline(report.threshold, color="#c99a2e", lw=1, ls="--")
    ax.set_ylim(0, 1.02); ax.set_xlim(0, report.duration)
    ax.set_xlabel("seconds"); ax.set_ylabel("music score")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="halal-audio-check", description="Flag music in video or audio files.")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--threshold", type=float, default=0.6, help="music score that counts as music (0-1, default 0.6)")
    ap.add_argument("--min-seconds", type=float, default=2.0, help="ignore flagged stretches shorter than this")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--timeline", metavar="PNG", help="save a score timeline (single file only)")
    ap.add_argument("--version", action="version", version=f"halal-audio-check {__version__}")
    a = ap.parse_args(argv)
    colour = sys.stdout.isatty() and not a.json
    paint = (lambda c, s: f"{c}{s}{RESET}") if colour else (lambda c, s: s)

    results, found, failed = [], False, False
    for f in a.files:
        try:
            r = check_file(f, threshold=a.threshold, min_segment=a.min_seconds)
        except Exception as e:  # noqa: BLE001 - report and carry on with the next file
            failed = True
            if a.json:
                results.append({"file": f, "error": str(e)})
            else:
                print(f"{f}: {paint(RED, 'error')} {e}")
            continue
        found |= not r.clean
        if a.json:
            results.append({"file": f, "duration": round(r.duration, 2), "clean": r.clean,
                            "music_seconds": round(r.music_seconds, 1),
                            "segments": [{"start": round(s.start, 1), "end": round(s.end, 1), "peak": round(s.peak, 2)}
                                         for s in r.segments]})
        else:
            if r.clean:
                print(f"{f}  {paint(GREEN, 'clean')}  {paint(DIM, f'no music found in {timestamp(r.duration)}')}")
            else:
                print(f"{f}  {paint(RED, 'MUSIC')}  {r.music_seconds:.0f}s flagged in {timestamp(r.duration)}")
                for s in r.segments:
                    print(f"  {timestamp(s.start)} - {timestamp(s.end)}  score {s.peak:.2f}")
        if a.timeline and len(a.files) == 1:
            timeline_png(r, a.timeline)
    if a.json:
        print(json.dumps(results if len(results) > 1 else results[0], indent=2))
    return 2 if failed else (1 if found else 0)


if __name__ == "__main__":
    raise SystemExit(main())
