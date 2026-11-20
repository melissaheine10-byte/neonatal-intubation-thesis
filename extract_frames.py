#!/usr/bin/env python3
"""Extract full-resolution video frames at a chosen rate or evenly spaced count.

Examples:
    python extract_frames.py "Intubation video.mp4" --fps 2
    python extract_frames.py "Intubation video.mp4" --count 50

Requires: pip install opencv-python-headless
Sampling uses the video's reported frame rate (constant-frame-rate videos).
"""

import argparse
import csv
import math
from pathlib import Path
import sys


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return number


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be an integer greater than zero")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="Input video path")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fps", type=positive_float,
                      help="Frames to extract per second; default: 1 (0.5 = one every 2 seconds)")
    mode.add_argument("--count", type=positive_int,
                      help="Total frames, evenly spaced from first to last; 1 extracts the first")
    parser.add_argument("--output", type=Path, help="Output folder; must be empty if it exists")
    parser.add_argument("--format", choices=("jpg", "png"), default="jpg",
                        help="Image format; PNG is lossless; default: jpg")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        parser.exit(1, "Install the dependency first: python -m pip install opencv-python-headless\n")

    if not args.video.is_file():
        parser.error(f"Video not found: {args.video}")
    output = args.output or args.video.with_name(args.video.stem + "_frames")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output must be an empty folder: {output}. Choose another --output folder.")

    cap = cv2.VideoCapture(str(args.video))
    try:
        if not cap.isOpened():
            raise RuntimeError("Could not open the video. Check its format and codec.")
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(source_fps) or source_fps <= 0 or total <= 0:
            raise RuntimeError("The video does not report a usable frame rate and frame count.")
        rate = args.fps if args.fps is not None else 1.0
        if args.count is not None and args.count > total:
            parser.error(f"Requested {args.count} frames, but the video reports only {total}.")
        if args.count is None and rate > source_fps:
            parser.error(f"Extraction rate cannot exceed the source rate ({source_fps:g} fps).")

        print(f"Video: {source_fps:g} fps, {total} frames, approximately {total/source_fps:.2f} seconds")
        output.mkdir(parents=True, exist_ok=True)

        def target_index(n):
            if args.count is not None:
                if args.count == 1:
                    return 0
                return round(n * (total - 1) / (args.count - 1))
            return round(n * source_fps / rate)

        saved = 0
        index = 0
        target = target_index(0)
        with (output / "frames.csv").open("w", newline="", encoding="utf-8") as manifest:
            writer = csv.writer(manifest)
            writer.writerow(["filename", "source_frame_index", "nominal_timestamp_seconds"])
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if index == target:
                    timestamp = index / source_fps
                    name = f"frame_{saved + 1:06d}_at_{timestamp:.3f}s.{args.format}"
                    settings = [cv2.IMWRITE_JPEG_QUALITY, 95] if args.format == "jpg" else []
                    if not cv2.imwrite(str(output / name), frame, settings):
                        raise RuntimeError(f"Could not save {name}")
                    writer.writerow([name, index, f"{timestamp:.6f}"])
                    saved += 1
                    if args.count is not None and saved == args.count:
                        break
                    target = target_index(saved)
                index += 1

        if saved == 0:
            raise RuntimeError("No frames could be decoded.")
        if args.count is not None and saved != args.count:
            raise RuntimeError(f"Video ended early: saved {saved} of {args.count} requested frames.")
        if args.count is None and index < total:
            raise RuntimeError(f"Video decoding ended early at frame {index} of {total}; saved {saved} frames.")
        print(f"Saved {saved} frames to: {output.resolve()}")
        print("Frame indices and approximate timestamps are recorded in frames.csv.")
    except (RuntimeError, OSError, cv2.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        cap.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
