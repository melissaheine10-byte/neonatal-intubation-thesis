"""Join DataTorch marks to source video frames and write a metadata-free CSV."""
import argparse
import csv
import json
import math
from pathlib import Path

FIELDS = ("frame_index", "time_s", "landmark_id", "landmark", "x_px", "y_px", "status")


def export(annotations, manifest, output):
    with manifest.open(newline="") as handle:
        frames = {row["filename"]: row for row in csv.DictReader(handle)}
    source = json.loads(annotations.read_text(encoding="utf-8"))
    images = {image["id"]: image for image in source["images"]}
    names = {category["id"]: category["name"] for category in source["categories"]}
    marks = {}
    for annotation in source["annotations"]:
        image_id, label_id = annotation["image_id"], annotation["category_id"]
        key = image_id, label_id
        image = images[image_id]
        if key in marks or label_id not in names:
            raise ValueError(f"Duplicate or unknown landmark at {key}")
        x, y, visibility = annotation["keypoints"]
        if visibility not in (0, 1, 2):
            raise ValueError(f"Unsupported visibility flag at {key}")
        if visibility and not (math.isfinite(x) and math.isfinite(y) and
                               0 <= x < image["width"] and 0 <= y < image["height"]):
            raise ValueError(f"Invalid marked position at {key}")
        marks[key] = (x, y, visibility)

    ordered = sorted(images.items(), key=lambda item: int(frames[item[1]["file_name"]]["source_frame_index"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(FIELDS)
        for image_id, image in ordered:
            frame = frames[image["file_name"]]
            for label_id, label in sorted(names.items()):
                x, y, visibility = marks.get((image_id, label_id), ("", "", 0))
                status = {0: "unannotated", 1: "manual_occluded", 2: "manual"}[visibility]
                writer.writerow([frame["source_frame_index"], frame["nominal_timestamp_seconds"],
                                 label_id, label, x if visibility else "", y if visibility else "", status])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export(args.annotations, args.manifest, args.output)
