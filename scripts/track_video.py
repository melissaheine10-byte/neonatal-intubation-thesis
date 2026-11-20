"""Track named DataTorch landmarks in source video coordinates with CoTracker3."""
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

MODEL_REVISION = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
WEIGHTS_URL = "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth"
COLOURS = ("#ff807c", "#ffba69", "#fce17c", "#b9ed79", "#54d9a0", "#62e6d8",
           "#74d5ff", "#93acff", "#b3a3ff", "#e19bff", "#f8a7d2", "#ff8bb8", "#ffffff")


@dataclass
class Marks:
    labels: dict
    manual: dict
    omitted: dict
    frame_indexes: list


def load_marks(path):
    labels, manual, omitted = {}, {}, {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame, point = int(row["frame_index"]), int(row["landmark_id"])
            name, status = row["landmark"], row["status"]
            if labels.setdefault(point, name) != name:
                raise ValueError(f"Inconsistent label for landmark {point}")
            key = frame, point
            if key in manual or key in omitted:
                raise ValueError(f"Duplicate manual row: {key}")
            if status == "manual":
                coordinates = row["x_px"], row["y_px"]
                if not all(math.isfinite(float(value)) for value in coordinates):
                    raise ValueError(f"Non-finite coordinate at {key}")
                manual[key] = coordinates
            elif status in ("unannotated", "manual_occluded"):
                omitted[key] = status
            else:
                raise ValueError(f"Unknown mark status: {status}")
    indexes = sorted({frame for frame, _ in manual.keys() | omitted.keys()})
    if not manual or not indexes or indexes[0] < 0:
        raise ValueError("Need manual marks at known source frame indexes")
    return Marks(dict(sorted(labels.items())), manual, omitted, indexes)


def read_video(path):
    with imageio.get_reader(str(path), format="FFMPEG", pixelformat="rgb24") as reader:
        fps = float(reader.get_meta_data()["fps"])
        frames = [frame for frame in reader]
    if not frames or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Could not read the source clip and its rate")
    height, width, channels = frames[0].shape
    if channels != 3 or any(frame.shape != (height, width, 3) for frame in frames):
        raise ValueError("Inconsistent video frame size")
    return frames, fps, width, height


def get_model(upstream, checkpoint, device):
    sys.path.insert(0, str(upstream.resolve()))
    from cotracker.predictor import CoTrackerPredictor
    return CoTrackerPredictor(checkpoint=str(checkpoint), window_len=60, v2=False).eval().to(device)


def estimate(model, clip, marks, seed_frame, device, backwards=False):
    # In the reversed clip the late anchor is the first video frame.
    seeds = [(point, *map(float, marks.manual[seed_frame, point]))
             for point in marks.labels if (seed_frame, point) in marks.manual]
    if not seeds:
        return {}
    ordered = clip[::-1] if backwards else clip
    video = torch.from_numpy(np.stack(ordered)).permute(0, 3, 1, 2)[None].to(device=device, dtype=torch.float32)
    query = torch.tensor([[[0, x, y] for _, x, y in seeds]], dtype=torch.float32, device=device)
    with torch.inference_mode():
        tracks, visibility = model(video, queries=query)
    coords = tracks[0].detach().cpu().numpy()
    visible = visibility[0].detach().cpu().numpy().reshape(len(clip), len(seeds))
    if backwards:
        coords, visible = coords[::-1], visible[::-1]
    return {point: (coords[:, i], visible[:, i]) for i, (point, _, _) in enumerate(seeds)}


def candidate(estimates, point, offset, width, height):
    if point not in estimates:
        return None
    coords, visible = estimates[point]
    x, y = coords[offset]
    if visible[offset] and np.isfinite((x, y)).all() and 0 <= x < width and 0 <= y < height:
        return float(x), float(y)
    return None


def holdout(marks, estimates, seed_frame, target_frame, offset, width, height):
    rows = []
    for point, name in marks.labels.items():
        actual = marks.manual.get((target_frame, point))
        if actual is None or point not in estimates:
            continue
        predicted = candidate(estimates, point, offset, width, height)
        rows.append({
            "landmark": name, "landmark_id": point, "seed_frame": seed_frame,
            "held_out_frame": target_frame, "visible_prediction": predicted is not None,
            "error_px": round(math.dist(tuple(map(float, actual)), predicted), 4) if predicted else None,
        })
    return rows


def make_rows(segment, forward, backward, marks, width, height, threshold):
    start, end, mode = segment
    for frame in range(start, end + 1):
        for point, name in marks.labels.items():
            before = candidate(forward, point, frame - start, width, height)
            after = candidate(backward, point, frame - start, width, height)
            disagreement = math.dist(before, after) if before is not None and after is not None else None
            coords = None
            if (frame, point) in marks.manual:
                status, coords = "manual", marks.manual[frame, point]
            elif (frame, point) in marks.omitted:
                status = marks.omitted[frame, point]
            elif disagreement is not None and disagreement > threshold:
                status = "disagreement"
            elif disagreement is not None:
                status = "model_two_seeds"
                coords = before if frame - start <= end - frame else after
            elif before is not None or after is not None:
                status, coords = "model_one_seed", before if before is not None else after
            else:
                status = "no_visible_prediction"
            yield {
                "frame_index": frame, "landmark_id": point, "landmark": name, "status": status,
                "x_px": format_coordinate(coords, 0, status), "y_px": format_coordinate(coords, 1, status),
                "seed_disagreement_px": "" if disagreement is None else round(disagreement, 4),
            }


def format_coordinate(coords, axis, status):
    if coords is None:
        return ""
    return coords[axis] if status == "manual" else round(coords[axis], 4)


def save_json(path, value):
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


def write_tracks(path, rows, fps):
    fields = ("frame_index", "time_s", "landmark_id", "landmark", "status", "x_px", "y_px", "seed_disagreement_px")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "time_s": f"{row['frame_index'] / fps:.6f}"})


def draw_clip(path, frames, rows, marks, fps):
    width, height = frames[0].shape[1], frames[0].shape[0]
    n_points = len(marks.labels)
    palette = {point: COLOURS[i % len(COLOURS)] for i, point in enumerate(marks.labels)}
    with imageio.get_writer(str(path), fps=fps, format="FFMPEG", codec="libx264", macro_block_size=2,
                            ffmpeg_log_level="error", output_params=["-movflags", "+faststart"]) as writer:
        for i, pixels in enumerate(frames):
            canvas = Image.new("RGB", (width + 190, height), "#152232")
            canvas.paste(Image.fromarray(pixels), (0, 0))
            draw = ImageDraw.Draw(canvas)
            draw.text((width + 10, 16), f"Frame {i}  |  {i / fps:.3f} s", fill="white")
            draw.text((width + 10, 44), "Solid: manual  Ring: model", fill="#cbd7e3")
            for j, row in enumerate(rows[i * n_points:(i + 1) * n_points]):
                point, name = row["landmark_id"], row["landmark"]
                if name == "TMJ":
                    name = "TMJ (jaw-bone tip)"
                colour = palette[point]
                label_y = 80 + j * 26
                draw.text((width + 10, label_y), f"{point:2d}  {name}", fill=colour)
                if row["x_px"] == "":
                    draw.text((width + 28, label_y + 12), "gap", fill="#a5b5c5")
                    continue
                x, y = float(row["x_px"]), float(row["y_px"])
                circle = (x - 4, y - 4, x + 4, y + 4)
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline="#152232", width=2)
                draw.ellipse(circle, outline=colour, width=2, fill=colour if row["status"] == "manual" else None)
                draw.text((x + 5, y - 12), str(point), fill=colour, stroke_width=2, stroke_fill="#152232")
            writer.append_data(np.asarray(canvas))


def validation_summary(rows):
    errors = [row["error_px"] for row in rows if row["error_px"] is not None]
    return {"comparisons": len(rows), "visible_predictions": len(errors),
            "median_error_px": round(median(errors), 4) if errors else None,
            "max_error_px": max(errors, default=None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--disagreement-px", type=float, default=8.0)
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or not math.isfinite(args.disagreement_px) or args.disagreement_px <= 0:
        parser.error("Threads and disagreement threshold must be positive")
    torch.set_num_threads(args.threads)
    marks = load_marks(args.manual)
    frames, fps, width, height = read_video(args.video)
    if marks.frame_indexes[-1] >= len(frames):
        raise ValueError("Manual index is outside the decoded clip")
    for key, (x, y) in marks.manual.items():
        if not (0 <= float(x) < width and 0 <= float(y) < height):
            raise ValueError(f"Manual coordinate outside decoded image at {key}")

    ends = marks.frame_indexes
    segments = [(left, right, "both") for left, right in zip(ends, ends[1:])]
    if ends[0] > 0:
        segments.insert(0, (0, ends[0], "backward"))
    if ends[-1] < len(frames) - 1:
        segments.append((ends[-1], len(frames) - 1, "forward"))
    if not segments:
        raise ValueError("Video needs at least two frames")
    pilot = next((segment for segment in segments if segment[:2] == (240, 300)), segments[0])
    ordered = [pilot] + [segment for segment in segments if segment != pilot]
    args.output.mkdir(parents=True, exist_ok=True)
    model = get_model(args.upstream, args.checkpoint, args.device)
    rows_by_key, validation, benchmarks = {}, [], []
    settings = {
        "model": "CoTracker3 Offline", "upstream_git_commit": MODEL_REVISION,
        "checkpoint_url": WEIGHTS_URL, "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "torch_version": torch.__version__, "device": args.device, "torch_threads": args.threads,
        "disagreement_threshold_px": args.disagreement_px,
        "source_width_px": width, "source_height_px": height, "fps_from_decoder": fps,
        "decoded_frames": len(frames), "manual_landmarks": len(marks.manual),
        "landmark_types": len(marks.labels),
    }
    for start, end, mode in ordered:
        clock = time.perf_counter()
        print(f"Tracking frames {start}–{end} ({mode})", flush=True)
        clip = frames[start:end + 1]
        forward = estimate(model, clip, marks, start, args.device) if mode != "backward" else {}
        backward = estimate(model, clip, marks, end, args.device, backwards=True) if mode != "forward" else {}
        elapsed = round(time.perf_counter() - clock, 2)
        metrics = {"first_frame": start, "last_frame": end, "inference_seconds": elapsed}
        benchmarks.append(metrics)
        print(f"Finished in {elapsed:.2f} seconds", flush=True)
        checks = holdout(marks, forward, start, end, end - start, width, height)
        checks += holdout(marks, backward, end, start, 0, width, height)
        validation.extend(checks)
        if (start, end, mode) == pilot:
            save_json(args.output / "benchmark.json", {**settings, **metrics,
                      "validation": validation_summary(checks), "held_out_marks": checks})
            if args.benchmark_only:
                return
        for row in make_rows((start, end, mode), forward, backward, marks, width, height, args.disagreement_px):
            rows_by_key[row["frame_index"], row["landmark_id"]] = row

    rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    assert len(rows) == len(frames) * len(marks.labels)
    assert {(r["frame_index"], r["landmark_id"]): (r["x_px"], r["y_px"])
            for r in rows if r["status"] == "manual"} == marks.manual
    write_tracks(args.output / "tracked_landmarks.csv", rows, fps)
    summary = {**settings, "segments": benchmarks, "row_counts": dict(Counter(row["status"] for row in rows)),
               "validation": validation_summary(validation), "held_out_marks": validation}
    save_json(args.output / "tracking_report.json", summary)
    draw_clip(args.output / "tracked_landmarks.mp4", frames, rows, marks, fps)
    print(f"Wrote {len(rows)} coordinate rows and {len(frames)} annotated frames", flush=True)


if __name__ == "__main__":
    main()
