# Wedding Photo Analyzer

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)

> **Automatically score, rank, and export your best wedding photos using computer vision — no manual culling required.**

Wedding photographers typically shoot 1,000–3,000 photos per event. This tool analyses every image across five quality dimensions and delivers a ranked shortlist in minutes, saving hours of manual review.

---

## Features

- **Recursive scanning** — processes entire folder trees, any nesting depth
- **5-metric quality scoring** — sharpness, exposure, resolution, contrast, face detection
- **Batch-normalised ranking** — scores are calibrated across your full collection, not arbitrary thresholds
- **Ranked CSV report** — sortable spreadsheet with per-metric breakdown for every photo
- **Auto-export** — top N photos copied to a `BEST_PRINTS/` folder, ready for client proofing
- **Progress bar** — live feedback with `tqdm` for large batches
- **Graceful error handling** — corrupted or unreadable files are skipped with a warning

---

## Scoring Model

Each photo receives a composite score from 0.0 to 1.0 using the following weighted metrics:

| Metric       | Weight | Method                                          |
|--------------|--------|-------------------------------------------------|
| Sharpness    | 35%    | Laplacian variance (higher = sharper edges)     |
| Exposure     | 25%    | Histogram analysis, penalises clipping          |
| Resolution   | 15%    | Megapixels, normalised to 24 MP ceiling         |
| Contrast     | 15%    | Pixel intensity standard deviation              |
| Face Detection | 10%  | Haar cascade frontal face detector              |

Sharpness and contrast are **min-max normalised** across the full batch before scoring, so the ranking is relative to the photos you provide.

---

## Requirements

- Python 3.10+
- `opencv-python`, `Pillow`, `numpy`, `tqdm`

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/wedding-photo-analyzer.git
cd wedding-photo-analyzer

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
# Analyse all photos in a folder (outputs CSV + copies top 50)
python analyze_photos.py /path/to/wedding/photos

# Export top 30 instead of 50
python analyze_photos.py /path/to/wedding/photos --top 30

# Save the CSV to a custom location
python analyze_photos.py /path/to/wedding/photos --output results/report.csv

# Score and rank only — skip copying to BEST_PRINTS
python analyze_photos.py /path/to/wedding/photos --no-copy
```

### Arguments

| Argument     | Description                                              | Default                          |
|--------------|----------------------------------------------------------|----------------------------------|
| `photo_dir`  | Root folder of wedding photos (required)                 | —                                |
| `--top N`    | Number of top photos to copy to `BEST_PRINTS/`           | `50`                             |
| `--output`   | CSV report output path                                   | `<photo_dir>/photo_scores.csv`   |
| `--no-copy`  | Skip the `BEST_PRINTS/` export step                      | `false`                          |

---

## Sample Output

### Terminal

```
=======================================================
  Wedding Photo Analyzer
=======================================================
  Folder : /Users/photographer/events/smith-wedding
  Top N  : 50
  Output : /Users/photographer/events/smith-wedding/photo_scores.csv
=======================================================

Found 1,847 images. Analysing...

Scoring: 100%|████████████████| 1847/1847 [03:12<00:00,  9.6 photo/s]

Ranked report saved  ->  /Users/photographer/events/smith-wedding/photo_scores.csv

Top 10 Photos
Rank   Score    Faces   MP      Filename
----------------------------------------------------------------------
1      0.8941   3       24.2    DSC_0847.jpg
2      0.8812   2       24.2    DSC_1203.jpg
3      0.8754   4       24.2    DSC_0612.jpg
4      0.8691   2       24.2    DSC_1455.jpg
5      0.8634   1       18.1    DSC_0391.jpg
...

Copying top 50 photos  ->  /Users/photographer/events/smith-wedding/BEST_PRINTS
Copying: 100%|████████████████| 50/50 [00:02<00:00, 22.4 file/s]
Done. 50 best-print photos ready in:
  /Users/photographer/events/smith-wedding/BEST_PRINTS
```

### CSV Report (`photo_scores.csv`)

| rank | filename        | final_score | sharpness_raw | exposure_score | resolution_mp | contrast_raw | face_count |
|------|-----------------|-------------|---------------|----------------|---------------|--------------|------------|
| 1    | DSC_0847.jpg    | 0.8941      | 4821.3        | 0.87           | 24.2          | 61.4         | 3          |
| 2    | DSC_1203.jpg    | 0.8812      | 4602.7        | 0.91           | 24.2          | 58.9         | 2          |
| 3    | DSC_0612.jpg    | 0.8754      | 4489.1        | 0.83           | 24.2          | 62.1         | 4          |

---

## Project Structure

```
wedding-photo-analyzer/
├── analyze_photos.py    # Main analysis script
├── requirements.txt     # Python dependencies
├── README.md
└── .gitignore
```

---

## License

Released under the [MIT License](LICENSE). Free to use, modify, and distribute.
