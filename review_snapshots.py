"""Sort the DataTorch export and create diagnostic overlays without changing points."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def timestamp(image):
    match = re.search(r"_at_([0-9.]+)s", image["file_name"])
    if not match:
        raise ValueError(f"Missing video timestamp: {image['file_name']}")
    return float(match.group(1))


def canonical(record):
    return json.dumps(record, sort_keys=True, ensure_ascii=False)


def text(im, label, xy, size=0.5, colour=(255, 255, 255)):
    (width, height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, size, 1)
    cv2.rectangle(im, (xy[0] - 2, xy[1] - height - 2),
                  (xy[0] + width + 2, xy[1] + baseline + 2), (25, 25, 25), -1)
    cv2.putText(im, label, xy, cv2.FONT_HERSHEY_SIMPLEX, size, colour, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, default=ROOT / "Snapshots_fixed-1.json")
    parser.add_argument("--frames", type=Path, default=ROOT / "Intubation video_frames")
    parser.add_argument("--output", type=Path, default=ROOT / "landmark_review")
    args = parser.parse_args()
    source, out = args.export, args.output
    dest = out / f"{source.stem}_sorted.json"
    if dest.resolve() == source.resolve():
        parser.error("Choose a different output folder to preserve the source export.")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    original = json.loads(source.read_text())
    data = json.loads(source.read_text())
    data["images"].sort(key=lambda im: (timestamp(im), im["id"]))
    order = {im["id"]: i for i, im in enumerate(data["images"])}
    cats = {c["id"]: c["name"] for c in data["categories"]}
    assert len(order) == len(data["images"])
    data["annotations"].sort(key=lambda a: (order[a["image_id"]], a["category_id"], a["id"]))
    pairs = Counter((a["image_id"], a["category_id"]) for a in data["annotations"])
    assert all(n == 1 for n in pairs.values()), "Duplicate landmark within image"
    for key in ("images", "annotations"):
        assert Counter(map(canonical, data[key])) == Counter(map(canonical, original[key]))
    assert {k: v for k, v in data.items() if k not in ("images", "annotations")} == {
        k: v for k, v in original.items() if k not in ("images", "annotations")}

    out.mkdir(exist_ok=True)
    (out / "overlays").mkdir(exist_ok=True)
    offsets = {1: (-44, -12), 2: (5, -22), 3: (18, 20), 4: (15, -5),
               5: (-42, 20), 6: (8, -12), 7: (-40, -16), 8: (15, 18),
               9: (-35, 22), 10: (10, 0), 11: (-35, -12), 12: (15, 24), 13: (-40, -8)}
    colours = {}
    for n, cid in enumerate(sorted(cats)):
        hsv = np.uint8([[[round(n * 179 / len(cats)), 200, 255]]])
        colours[cid] = tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
    panels, rows = [], []
    for im in data["images"]:
        frame = cv2.imread(str(args.frames / im["file_name"]))
        if frame is None:
            raise ValueError(f"Cannot read {im['file_name']}")
        assert frame.shape[:2] == (im["height"], im["width"])
        annotations = {a["category_id"]: a for a in data["annotations"] if a["image_id"] == im["id"]}
        missing = [cid for cid in sorted(cats) if cid not in annotations]
        scale = 2
        overlay = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        for cid in sorted(cats):
            a = annotations.get(cid)
            if a is None:
                rows.append([im["file_name"], timestamp(im), im["id"], cid, cats[cid], "", "", "", "not_annotated"])
                continue
            x, y, visible = a["keypoints"]
            assert np.isfinite([x, y, visible]).all()
            assert 0 <= x < im["width"] and 0 <= y < im["height"]
            rows.append([im["file_name"], timestamp(im), im["id"], cid, cats[cid], x, y, visible, "annotated"])
            if not visible:
                continue
            p = (round(x * scale), round(y * scale))
            dx, dy = offsets[cid]
            q = (p[0] + dx * scale, p[1] + dy * scale)
            cv2.line(overlay, p, q, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.line(overlay, p, q, colours[cid], 1, cv2.LINE_AA)
            cv2.drawMarker(overlay, p, (0, 0, 0), cv2.MARKER_CROSS, 17, 5)
            cv2.drawMarker(overlay, p, colours[cid], cv2.MARKER_CROSS, 15, 2)
            text(overlay, str(cid), q, 0.9, colours[cid])
        panel = np.full((1900, 956, 3), 30, dtype=np.uint8)
        panel[70:1770] = overlay
        text(panel, f"{im['file_name']} | image ID {im['id']}", (15, 35), 0.7)
        text(panel, "Missing IDs: " + (", ".join(map(str, missing)) or "none"), (15, 1810), 0.7)
        text(panel, "Cross centre = exact exported coordinate; labels offset for clarity", (15, 1855), 0.55)
        filename = out / "overlays" / (Path(im["file_name"]).stem + "_review.png")
        assert cv2.imwrite(str(filename), panel)
        panels.append(cv2.resize(panel, (478, 950), interpolation=cv2.INTER_AREA))

    sheet = np.full((2140, 1912, 3), 30, dtype=np.uint8)
    for i, panel in enumerate(panels):
        r, c = divmod(i, 4)
        sheet[r * 950:(r + 1) * 950, c * 478:(c + 1) * 478] = panel
    text(sheet, "LANDMARK KEY | chronological order: left to right, top row then bottom row", (20, 1935), 0.7)
    for i, cid in enumerate(sorted(cats)):
        r, c = divmod(i, 4)
        label = f"{cid}: {cats[cid]}" + (" (jaw-bone tip)" if cid == 13 else "")
        text(sheet, label, (20 + c * 478, 1980 + r * 42), 0.55, colours[cid])
    assert cv2.imwrite(str(out / "comparison_sheet.png"), sheet)
    with (out / "landmarks.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "time_seconds_from_filename", "image_id", "category_id", "landmark", "x_pixels", "y_pixels", "export_visibility_flag", "annotation_status"])
        writer.writerows(rows)

    dest.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    saved = json.loads(dest.read_text())
    assert saved == data
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    verification = {"original_sha256": before, "original_unchanged": True,
                    "images": len(data["images"]), "categories": len(cats),
                    "annotations": len(data["annotations"]), "missing_positions": sum(r[-1] == "not_annotated" for r in rows),
                    "image_and_annotation_records_preserved": True, "other_fields_preserved": True,
                    "duplicate_landmarks": 0, "all_annotated_coordinates_in_bounds": True,
                    "sorted_video_times_seconds": [timestamp(im) for im in data["images"]]}
    (out / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
