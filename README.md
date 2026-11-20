# Neonatal intubation thesis

This project explores a mechanical newborn jaw and neck model for intubation simulation. The design goal is to make jaw opening, sliding, and resistance adjustable, and to study how jaw motion interacts with neck position. The intended applications include representing differences between preterm and full-term newborns in a simulator.

The repository documents the mechanical design and the video analysis used to study how parts of a training manikin move. The motion data are two-dimensional image coordinates from a manikin demonstration.

| Area | What is included |
| --- | --- |
| Mechanical design | [Design overview and literature references](docs/mechanical-design.md) |
| Manual annotation | [DataTorch workflow and tracking method](docs/video-analysis.md) and [manual landmark coordinates](data/manual_landmarks.csv) |
| Tracking results | [Coordinates for every video frame](data/tracked_landmarks.csv) and [run findings](docs/video-analysis.md#run-results) |
| Reproducibility | [Frame extraction](extract_frames.py), [snapshot review](review_snapshots.py), [snapshot comparison](compare_landmarks.py), [annotation export](scripts/export_manual.py), and [CoTracker inference](scripts/track_video.py) |

The manual dataset covers **13 landmarks across eight snapshots**. It contains **98 visible marks** and six unannotated positions. The field named **TMJ** identifies the tip of the manikin’s jaw member. The mark at approximately **9.995 seconds** was added after the first annotation review.

CoTracker processed all **468 video frames**. The public tables preserve the manual coordinates, identify model suggestions, and leave flagged gaps blank. The [validation report](data/tracking_report.json) records the held-out checks, including large errors on some lip and chin predictions. The source footage and annotated video are stored locally.
