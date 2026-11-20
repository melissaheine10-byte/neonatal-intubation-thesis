"""Compare manually marked DataTorch snapshots using NumPy, Pillow and Matplotlib.

Run `python3 compare_landmarks.py` from any folder, or specify an export with
`--export PATH`. Coordinates retain the image convention: x right, y down.
The torso view pins Shoulder and Chest to their positions in a reference
snapshot with a 2D similarity transform. This is an image-space comparison.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
PALETTE = ("#ff807c", "#ffba69", "#fce17c", "#b9ed79", "#54d9a0", "#62e6d8",
           "#74d5ff", "#93acff", "#b3a3ff", "#e19bff", "#f8a7d2", "#ff8bb8", "#fcfcfc")
INK = "#172638"
SUBTLE = "#435568"


def video_time(image):
    match = re.search(r"_at_(\d+(?:\.\d+)?)s\.", image["file_name"])
    if not match:
        raise ValueError(f"Missing video timestamp in {image['file_name']}")
    return float(match[1])


def load_export(path):
    payload = json.loads(path.read_bytes())
    payload["images"].sort(key=video_time)
    images = payload["images"]
    categories = sorted(payload["categories"], key=lambda c: c["id"])
    times = np.array([video_time(im) for im in images])
    image_indexes = {im["id"]: i for i, im in enumerate(images)}
    point_indexes = {cat["id"]: i for i, cat in enumerate(categories)}
    if len(image_indexes) != len(images) or len(set(times)) != len(times):
        raise ValueError("Duplicate image IDs or video timestamps")
    if len(point_indexes) != len(categories):
        raise ValueError("Duplicate landmark IDs")
    payload["annotations"].sort(key=lambda a: (image_indexes[a["image_id"]], a["category_id"], a["id"]))
    points = np.full((len(images), len(categories), 2), np.nan)
    marks = {}
    for annotation in payload["annotations"]:
        i = image_indexes[annotation["image_id"]]
        j = point_indexes[annotation["category_id"]]
        if (i, j) in marks:
            raise ValueError(f"Duplicate annotation for {images[i]['file_name']} and {categories[j]['name']}")
        x, y, visibility = annotation["keypoints"]
        marks[i, j] = (x, y, visibility)
        if visibility == 2:
            if not (0 <= x < images[i]["width"] and 0 <= y < images[i]["height"]):
                raise ValueError(f"Invalid coordinate in annotation {annotation['id']}")
            points[i, j] = (x, y)
    return payload, times, categories, points, marks


def torso_transforms(points, shoulder, chest, reference):
    origins = points[:, shoulder]
    vectors = points[:, chest] - origins
    lengths = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(vectors).all() or np.any(lengths <= 0):
        raise ValueError("Every snapshot needs distinct, visible Shoulder and Chest points")
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    rotation = angles[reference] - angles
    scales = lengths[reference] / lengths
    matrices = np.array([s * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                         for s, a in zip(scales, rotation)])
    translations = origins[reference] - np.einsum("fij,fj->fi", matrices, origins)
    aligned = np.einsum("fij,fkj->fki", matrices, points) + translations[:, None, :]
    np.testing.assert_allclose(aligned[:, [shoulder, chest]],
                               np.broadcast_to(points[reference, [shoulder, chest]], (len(points), 2, 2)), atol=1e-8)
    return aligned, matrices, translations, lengths, scales, np.degrees(rotation)


def read_frames(images, folder, matrices, translations):
    frames, aligned_frames = [], []
    for record, matrix, offset in zip(images, matrices, translations):
        with Image.open(folder / record["file_name"]) as source:
            if source.size != (record["width"], record["height"]):
                raise ValueError(f"Image size mismatch for {record['file_name']}")
            frame = source.convert("RGB")
        inv = np.linalg.inv(matrix)
        # Pillow applies an inverse map in pixel-corner coordinates.
        half = np.array([0.5, 0.5])
        inverse_offset = half - inv @ (offset + half)
        coefficients = tuple(np.column_stack((inv, inverse_offset)).ravel())
        registered = frame.transform(frame.size, Image.Transform.AFFINE, coefficients,
                                     resample=Image.Resampling.BICUBIC, fillcolor=(22, 33, 47))
        frames.append(frame)
        aligned_frames.append(registered)
    return frames, aligned_frames


def style_figure(fig, title, note):
    fig.patch.set_facecolor("#f8fafc")
    fig.text(0.045, 0.967, title, fontsize=18, weight="bold", color=INK, va="top")
    fig.text(0.045, 0.923, note, fontsize=10, color=SUBTLE, va="top")


def image_crop(raw, aligned):
    both = np.concatenate((raw, aligned), axis=0)
    lo = np.nanmin(both, axis=(0, 1)) - [24, 30]
    hi = np.nanmax(both, axis=(0, 1)) + [24, 30]
    return lo, hi


def draw_overlay(ax, frame, coords, end, categories, crop, anchors, show_anchor_trails=False):
    ax.set_facecolor("#16212f")
    ax.imshow(frame, interpolation="antialiased")
    for j, cat in enumerate(categories):
        trail = coords[:end + 1, j]
        colour = PALETTE[j % len(PALETTE)]
        if j not in anchors or show_anchor_trails:
            ax.plot(*trail.T, color="#152234", alpha=0.72, lw=3)
            ax.plot(*trail.T, color=colour, alpha=0.9, lw=1.3, linestyle=(0, (2, 2)))
        point = trail[-1]
        if np.isfinite(point).all():
            ax.scatter(*point, s=28 if j in anchors else 19, marker="D" if j in anchors else "o",
                       c=colour, edgecolors="#16212f", linewidths=0.8, zorder=5)
    lo, hi = crop
    ax.set(xlim=(lo[0], hi[0]), ylim=(hi[1], lo[1]), aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])


def display_name(cat):
    return "TMJ (jaw-bone tip)" if cat["name"] == "TMJ" else cat["name"]


def legend_handles(categories, anchors):
    return [Line2D([], [], marker="D" if j in anchors else "o", linestyle="none",
                   markersize=6, markerfacecolor=PALETTE[j % len(PALETTE)], markeredgecolor=INK,
                   label=display_name(cat)) for j, cat in enumerate(categories)]


def write_replay(out, times, categories, raw, aligned, frames, registered, reference, anchors):
    crop = image_crop(raw, aligned)
    gif_frames = []
    for i, time in enumerate(times):
        fig, axes = plt.subplots(1, 2, figsize=(9.8, 5.5), dpi=105)
        style_figure(fig, f"Landmark replay  |  {time:.3f} s",
                     f"Snapshot {i + 1} of {len(times)}  •  Dots are current marks. Dashed trails link successive marked snapshots.")
        fig.subplots_adjust(left=.045, right=.955, bottom=.24, top=.85, wspace=.075)
        draw_overlay(axes[0], frames[i], raw, i, categories, crop, anchors, show_anchor_trails=True)
        draw_overlay(axes[1], registered[i], aligned, i, categories, crop, anchors)
        axes[0].set_title("Original image coordinates", loc="left", fontsize=11, color=INK)
        axes[1].set_title(f"Torso aligned to {times[reference]:.3f} s", loc="left", fontsize=11, color=INK)
        missing = [display_name(cat) for j, cat in enumerate(categories) if not np.isfinite(raw[i, j]).all()]
        fig.text(.045, .215, "Missing marks: " + (", ".join(missing) or "none"), fontsize=8.5, color=SUBTLE)
        fig.legend(handles=legend_handles(categories, anchors), loc="lower left", bbox_to_anchor=(.035, .02),
                   ncol=4, frameon=False, fontsize=8, labelcolor=INK, columnspacing=1.7)
        fig.canvas.draw()
        rgb = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
        gif_frames.append(rgb.quantize(colors=256))
        plt.close(fig)
    durations = [round(1000 * delta) for delta in np.diff(times)]
    durations.append(1800)
    gif_frames[0].save(out / "snapshot_replay.gif", save_all=True, append_images=gif_frames[1:],
                       duration=durations, loop=0, disposal=2)


def write_overlay_sheet(out, times, categories, raw, aligned, registered, anchors, reference):
    columns = min(4, len(times))
    rows = (len(times) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(12, 3.1 * rows + 1.5), squeeze=False)
    style_figure(fig, "Chest and shoulder aligned snapshots",
                 f"Reference image at {times[reference]:.3f} s  •  Dots mark observations. Dashed trails stop at missing snapshots.")
    fig.subplots_adjust(left=.045, right=.955, bottom=.16, top=.86, wspace=.075, hspace=.14)
    crop = image_crop(raw, aligned)
    for i, ax in enumerate(axes.flat):
        if i >= len(times):
            ax.set_axis_off()
            continue
        draw_overlay(ax, registered[i], aligned, i, categories, crop, anchors)
        ax.set_title(f"{i + 1:02d}   {times[i]:.3f} s", loc="left", fontsize=10, weight="bold", color=INK)
    fig.legend(handles=legend_handles(categories, anchors), loc="lower left", bbox_to_anchor=(.035, .025),
               ncol=5, frameon=False, fontsize=9, labelcolor=INK, columnspacing=1.7)
    fig.savefig(out / "aligned_snapshots.png", dpi=180)
    plt.close(fig)


def write_trajectories(out, times, categories, aligned, anchors):
    columns = 4
    rows = (len(categories) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows), squeeze=False)
    style_figure(fig, "Landmark movement relative to the torso",
                 "Colour shows video time. Dashed segments link successive observations. Each panel has its own axis limits.")
    fig.subplots_adjust(left=.09, right=.95, bottom=.115, top=.885, wspace=.55, hspace=.62)
    time_colour = Normalize(times[0], times[-1])
    for j, ax in enumerate(axes.flat):
        if j >= len(categories):
            ax.set_axis_off()
            continue
        coords = aligned[:, j]
        observed = np.isfinite(coords).all(axis=1)
        ax.set_title(display_name(categories[j]), fontsize=10, loc="left", weight="bold", color=INK)
        if j in anchors or not observed.any():
            ax.set_axis_off()
            label = "Fixed reference point" if j in anchors else "No visible observations"
            ax.text(.5, .5, label, transform=ax.transAxes, ha="center", color=SUBTLE, fontsize=10)
            continue
        ax.plot(*coords.T, linestyle="--", linewidth=1.15, color="#77899c", zorder=1)
        ax.scatter(*coords[observed].T, c=times[observed], cmap="viridis", norm=time_colour,
                   s=40, edgecolors="white", linewidths=.8, zorder=2)
        ax.set_aspect("equal", adjustable="datalim")
        ax.invert_yaxis()
        ax.tick_params(labelsize=7, colors=SUBTLE, length=2)
        ax.locator_params(axis="both", nbins=4)
        ax.grid(color="#e2e8f0", linewidth=.6)
        for spine in ax.spines.values():
            spine.set_color("#bac7d4")
    fig.text(.5, .081, "Horizontal image position (reference pixels)", ha="center", color=SUBTLE, fontsize=10)
    fig.text(.027, .50, "Vertical image position (reference pixels, positive down)", va="center",
             rotation="vertical", color=SUBTLE, fontsize=10)
    cax = fig.add_axes([.32, .042, .36, .012])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=time_colour, cmap="viridis"), cax=cax,
                       orientation="horizontal", ticks=times)
    bar.set_ticklabels([f"{time:.3f}" for time in times])
    bar.ax.tick_params(labelsize=7)
    bar.set_label("Video time (s)", fontsize=9, color=SUBTLE)
    fig.savefig(out / "landmark_trajectories.png", dpi=180)
    plt.close(fig)


def write_coordinates(out, payload, categories, times, marks, aligned, lengths, scales, rotations):
    with (out / "landmark_coordinates.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "image", "landmark_id", "landmark", "status", "visibility_flag",
                         "image_x_px", "image_y_px", "torso_aligned_x_reference_px", "torso_aligned_y_reference_px",
                         "chest_shoulder_input_distance_px", "alignment_scale", "alignment_rotation_deg_image"])
        for i, (record, time) in enumerate(zip(payload["images"], times)):
            for j, cat in enumerate(categories):
                mark = marks.get((i, j))
                status = "unannotated"
                coords = ["", "", "", "", ""]
                if mark is not None:
                    x, y, flag = mark
                    status = {0: "not_labelled", 1: "labelled_occluded", 2: "observed"}[flag]
                    coords[:3] = [flag, x if flag else "", y if flag else ""]
                    if flag == 2:
                        coords[3:] = [f"{value:.6f}" for value in aligned[i, j]]
                writer.writerow([f"{time:.3f}", record["file_name"], cat["id"], cat["name"], status, *coords,
                                 f"{lengths[i]:.6f}", f"{scales[i]:.8f}", f"{rotations[i]:.6f}"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, default=ROOT / "Snapshots_fixed-1.json")
    parser.add_argument("--frames", type=Path, default=ROOT / "Intubation video_frames")
    parser.add_argument("--output", type=Path, default=ROOT / "landmark_movement")
    parser.add_argument("--reference", type=float, default=11.994, help="Reference snapshot's video time in seconds")
    args = parser.parse_args()
    payload, times, categories, raw, marks = load_export(args.export)
    name_indexes = {cat["name"]: i for i, cat in enumerate(categories)}
    shoulder, chest = (name_indexes[name] for name in ("Shoulder", "Chest"))
    indices = np.flatnonzero(np.isclose(times, args.reference, rtol=0, atol=0.0005))
    if len(indices) != 1:
        parser.error(f"Reference must match a snapshot time. Available times: {times.tolist()}")
    reference = int(indices[0])
    result = torso_transforms(raw, shoulder, chest, reference)
    aligned, matrices, translations, lengths, scales, rotations = result
    frames, registered = read_frames(payload["images"], args.frames, matrices, translations)
    anchors = (shoulder, chest)

    args.output.mkdir(parents=True, exist_ok=True)
    sorted_path = args.output / f"{args.export.stem}_sorted.json"
    if sorted_path.resolve() == args.export.resolve():
        parser.error("Choose a different output folder to preserve the source export")
    sorted_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    original = json.loads(args.export.read_bytes())
    round_trip = json.loads(sorted_path.read_bytes())
    for key in ("images", "annotations"):
        original[key].sort(key=lambda record: record["id"])
        round_trip[key].sort(key=lambda record: record["id"])
    if round_trip != original:
        raise RuntimeError("Sorted export differs from its source records")

    write_coordinates(args.output, payload, categories, times, marks, aligned, lengths, scales, rotations)
    write_overlay_sheet(args.output, times, categories, raw, aligned, registered, anchors, reference)
    write_trajectories(args.output, times, categories, aligned, anchors)
    write_replay(args.output, times, categories, raw, aligned, frames, registered, reference, anchors)
    print(f"Saved {len(times)} snapshots and {len(marks)} point annotations to {args.output}")
    print(f"Chest-to-shoulder distance: {min(lengths):.1f}–{max(lengths):.1f} px")


if __name__ == "__main__":
    main()
