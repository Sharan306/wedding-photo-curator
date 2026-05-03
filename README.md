# Wedding Photo Curator

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue?style=flat&logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-brightgreen?style=flat&logo=opencv)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff69b4?style=flat&logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](LICENSE)

---

## Overview

**Wedding Photo Curator** is a quality-first photo curation tool that automatically selects only the best photos from large collections. Using hard rejection rules and diversity filtering, it embodies the philosophy: **"20 perfect photos > 100 average ones."**

Instead of ranked scoring that returns hundreds of mediocre photos, this tool applies strict quality criteria and intelligently filters duplicates to deliver a curated portfolio of truly exceptional moments.

Perfect for **wedding albums**, **event photography**, **travel collections**, and any scenario where quality over quantity is paramount.

---

## Philosophy

Most photo ranking systems try to score every photo and return "the top 50". This approach often results in:
- Dozens of similar shots (same pose, different angles)
- Marginal quality variations between ranked photos
- Overwhelming choices instead of clear selections

**Wedding Photo Curator** takes a different approach:

1. **Hard Rejections** — Automatically exclude photos that don't meet minimum standards
   - Blurry (sharpness < 100)
   - Poorly lit (lighting quality < 0.3)
   - Low resolution (< 20 MP)
   - No faces detected

2. **Diversity Filtering** — Remove perceptual duplicates and similar moments
   - Segment-based max-1-per-20-photos rule
   - 70% perceptual hash similarity threshold
   - No repeated poses

3. **Quality Scoring** — Score only the "worthy" candidates
   - Sharpness (25%)
   - Face quality: expression, eye contact, focus (30%)
   - Lighting quality (20%)
   - Composition and framing (15%)
   - Uniqueness vs already selected (10%)

**Result:** A small, diverse, genuinely excellent portfolio ready for your album.

---

## Features

- **Hard Rejection Rules** — Automatic exclusion of blurry, poorly lit, low-res, or no-face photos
- **Perceptual Deduplication** — Remove near-identical consecutive shots automatically
- **Segment-Based Diversity** — Limit selection to 1 photo per 20-photo segment
- **Quality-First Scoring** — 5-factor weighted score for worthy candidates only
- **Interactive Web Gallery** — Browse curated photos with detailed metrics
- **Fast Analysis** — Process 1000+ photos in under 5 minutes
- **JSON Caching** — Instant re-runs on cached analyses
- **CSV Export** — Detailed results with all metrics
- **Streamlit Cloud Ready** — Deployed serverless with no desktop dependencies
- **Fully Offline & Free** — No cloud APIs, no subscriptions, runs on CPU
- **Production UI** — Beautiful Streamlit app with real-time progress
- **Highly Customizable** — Adjust thresholds, weights, and criteria

---

## How It Works

### Hard Rejection Pipeline

Each photo passes through 4 quality gates:

| Gate | Metric | Min Threshold | Reason |
|------|--------|---------------|--------|
| 1. Sharpness | Laplacian variance | 100.0 | Rejects blurry/soft focus |
| 2. Lighting | Histogram analysis (0-1) | 0.3 | Rejects dark/overexposed/harsh |
| 3. Resolution | Megapixels | 20.0 | Rejects low-resolution prints |
| 4. Faces | Haar cascade count | 1+ | Rejects photos with no faces |

**Any photo failing ANY gate is automatically rejected.**

### Diversity Filtering

After hard rejections, remaining photos are filtered for diversity:

- **Perceptual Hashing** — Computes pHash for each photo
- **Similarity Check** — Photos > 70% similar to already-selected are skipped
- **Segment Rule** — Max 1 photo per 20-photo segment of the original sequence
- **Result** — No burst duplicates, diverse moments

### Quality Scoring

Only photos passing both gates are scored using:

```
Final Score = 0.25 × Sharpness 
            + 0.30 × Face Quality
            + 0.20 × Lighting
            + 0.15 × Composition
            + 0.10 × Uniqueness
```

Rankings are based solely on quality, not quantity.

---

## Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/wedding-photo-curator.git
   cd wedding-photo-curator
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

**Option 1: Web Interface (Recommended)**
```bash
streamlit run app.py
```
Then:
1. Enter your photo directory path
2. Click **"Analyze & Curate Photos"**
3. View the results gallery
4. Export curated photos to BEST_PRINTS/

**Option 2: Command-Line Batch Analysis**
```bash
python analyze_photos.py /path/to/wedding/photos
```

Outputs:
- `photo_curation.csv` — Ranked curated photos with metrics
- `BEST_PRINTS/` folder — All curated photos ready for album

**Advanced Options:**
```bash
python analyze_photos.py /photos --output-csv results.csv --no-copy
python analyze_photos.py /photos --no-cache  # Re-analyze from scratch
```

---

## Usage Guide

### Via Streamlit Web App

1. **Launch the app:**
   ```bash
   streamlit run app.py
   ```

2. **Enter your photo folder path:**
   - Paste the folder path in the sidebar input field

3. **Start curation:**
   - Click "Analyze & Curate Photos"
   - Watch progress as hard rejections and diversity filtering are applied

4. **Review summary:**
   - Total photos analyzed
   - Selected curated count
   - Selection rate percentage

5. **Browse the curated gallery:**
   - Photos ranked by quality score
   - Each card shows: rank, thumbnail, filename
   - Photos are guaranteed to pass all quality gates

6. **Export:**
   - Click "Copy to BEST_PRINTS" to copy all curated photos
   - Click "Download CSV Report" for detailed metrics

### Via Command Line

```bash
python analyze_photos.py /path/to/photos
```

The `BEST_PRINTS` folder will be created with all curated photos.

For batch processing with caching:
```bash
python analyze_photos.py /photos --no-cache  # Skip cache, re-analyze
```

---

## Tech Stack

| Component | Purpose |
|-----------|---------|
| **Python 3.8+** | Core language |
| **OpenCV** | Image processing, face detection, sharpness/lighting analysis |
| **Pillow (PIL)** | Image loading and format conversion |
| **ImageHash** | Perceptual hashing for duplicate detection |
| **NumPy** | Numerical computations and normalization |
| **Streamlit** | Interactive web UI |
| **tqdm** | Progress bars for batch operations |

---

## Use Cases

- **Wedding Photography Studios** — Curate albums from 1000+ photos
- **Event Photographers** — Select diverse, high-quality shots from events
- **Travel Photographers** — Curate print-worthy travel collections
- **Photo Archives** — Organize and select from personal collections
- **Photography Educators** — Teach quality assessment principles

---

## Sample Output

### CSV Report (`photo_curation.csv`)

```
rank,filename,final_score,sharpness,lighting,face_quality,composition,resolution,num_faces,path
1,IMG_0847.jpg,0.8124,285.5,0.92,0.95,0.88,24.3,2,/photos/IMG_0847.jpg
2,IMG_0891.jpg,0.7856,275.2,0.88,0.92,0.85,21.5,1,/photos/IMG_0891.jpg
3,IMG_0742.jpg,0.7642,265.8,0.85,0.90,0.82,18.0,1,/photos/IMG_0742.jpg
```

### Directory Structure After Export

```
wedding-photos/
├── IMG_0001.jpg
├── IMG_0015.jpg
├── ... (all 1287 original photos)
└── BEST_PRINTS/
    ├── IMG_0847.jpg          ← Curated: passed all quality gates
    ├── IMG_0891.jpg             (no duplicates or similar moments)
    ├── IMG_0742.jpg
    └── ... (23 more curated shots)
```

---

## Configuration & Customization

### Hard Rejection Thresholds

Edit constants in `analyze_photos.py`:

```python
SHARPNESS_MIN = 100.0         # Laplacian variance threshold
LIGHTING_MIN = 0.3            # Histogram quality (0-1 scale)
RESOLUTION_MIN = 20.0         # Minimum megapixels
FACE_DETECT_MIN = 1           # Minimum faces required
```

### Scoring Weights

Customize how photos are ranked (for those passing hard rejections):

```python
SCORE_WEIGHTS = {
    "sharpness": 0.25,        # Edge clarity
    "face_quality": 0.30,     # Face detection quality
    "lighting": 0.20,         # Overall lighting quality
    "composition": 0.15,      # Framing and edges
    "uniqueness": 0.10,       # Diversity vs selected
}
```

### Diversity Rules

Adjust how similar/duplicate photos are handled:

```python
SEGMENT_SIZE = 20             # Max 1 photo per segment
PERCEPTUAL_HASH_SIMILARITY = 70.0  # % threshold for "duplicate"
```

---

## Performance

- **Speed**: Process 1000+ photos in < 5 minutes on modern CPU
- **Memory**: Minimal footprint, all operations in-memory
- **Caching**: JSON cache for instant re-runs
- **Parallelization**: Ready for multi-core optimization

Typical performance:
- 1000 photos: 2-3 minutes
- 2000 photos: 4-5 minutes
- 5000 photos: 10-12 minutes

---

## Troubleshooting

### No photos selected
- Check that your photos have detectable faces
- Verify resolution is >= 20 MP
- Ensure photos aren't blurry (sharpness >= 100)
- Check lighting isn't too dark/bright (lighting >= 0.3)

### Too few photos selected
- Lower the hard rejection thresholds if needed
- Adjust diversity thresholds (SEGMENT_SIZE, PERCEPTUAL_HASH_SIMILARITY)

### Analysis is slow
- Ensure no other CPU-intensive processes are running
- Check that images aren't corrupted
- Monitor disk I/O during analysis

---

## Contributing

Contributions are welcome! Areas for enhancement:

- [ ] Color balance and saturation metrics
- [ ] Emotion detection for candid scoring
- [ ] Motion blur detection
- [ ] Multi-core parallelization
- [ ] Advanced filtering by date/location
- [ ] Dark theme toggle

**To contribute:**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Technical Highlights

This project demonstrates advanced computer vision techniques:

- **Perceptual Hashing** — Efficient near-duplicate detection using pHash
- **Face Detection** — Haar Cascade classifiers for real-time face recognition
- **Image Processing** — OpenCV Laplacian edge detection, histogram analysis
- **Quality Metrics** — Multi-factor assessment combining multiple vision approaches
- **Data Pipeline** — Efficient batch processing with caching
- **Modern Web Development** — Streamlit with real-time progress
- **Production Code** — Clean architecture, error handling, logging, type hints
- **CPU-Optimized** — No GPU required, runs on any machine

---

**Built with love for photographers and CV enthusiasts.**
