# 💍 Wedding Photo Curator

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue?style=flat&logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-brightgreen?style=flat&logo=opencv)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff69b4?style=flat&logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](LICENSE)

---

## 📋 Overview

**Wedding Photo Curator** is an AI-powered application that automatically analyzes and selects the best photos from large collections using computer vision and machine learning techniques. Instead of manually reviewing hundreds (or thousands) of wedding photos, photographers and studios can now leverage intelligent scoring metrics to curate their best prints in minutes.

Perfect for **wedding albums**, **event photography**, **travel collections**, and any scenario where rapid photo selection and quality assessment is needed.

---

## ✨ Features

- 🤖 **AI-Powered Semantic Scoring** — CLIP vision-language model understands photo content (not just technical metrics)
- 🎯 **Hybrid Intelligence** — Combines CLIP semantics (50%) with OpenCV metrics (50%)
- 🖼️ **Interactive Visual Gallery** — Browse ranked photos with approve/reject controls
- 💡 **AI Insights Per Photo** — "AI says: a candid emotional wedding moment" explains what CLIP sees
- 📊 **Detailed Metrics Display** — CLIP score, sharpness, exposure, resolution, face detection
- 💾 **Smart Curation** — Threshold-based selection ensures diverse portfolio, no burst duplicates
- 🤝 **Perceptual Deduplication** — Removes near-identical consecutive shots automatically
- 📁 **Native Folder Picker** — macOS Finder integration for seamless file selection
- 🚀 **Fully Offline & Free** — No cloud APIs, no subscriptions, runs on CPU or GPU
- 🎨 **Production UI** — Beautiful Streamlit app with real-time progress and responsive design
- ⚙️ **Highly Customizable** — Adjust weights, prompts, thresholds to your workflow

---

## 🧠 How It Works

Wedding Photo Curator uses a **hybrid AI approach** combining semantic understanding with computer vision:

### Scoring Pipeline

Each photo is evaluated using:

**1. CLIP Semantic Analysis (50%)**
- Evaluates against positive prompts: "beautiful well-lit portrait", "candid emotional wedding moment", "sharp focused photo", "flattering angle"
- Penalizes negatives: "blurry dark poorly composed photo"
- Returns interpretable AI insight: *"AI says: a candid emotional wedding moment"*

**2. Computer Vision Metrics (50%)**
- **Sharpness** (20%) — Laplacian variance for edge clarity
- **Exposure** (15%) — Histogram analysis, penalizes clipping
- **Resolution** (10%) — Megapixels relative to print standards
- **Face Detection** (5%) — Rewards portraits and candids

### Selection Strategy

- **Threshold-Based** (when CLIP available) — Selects all photos scoring ≥0.25
- **Diversity Filtering** — Perceptual hashing removes near-duplicates, keeping highest-scoring from each burst
- **Result** — Portfolio of diverse, high-quality moments (not 50 variations of the same pose)

---

## 🚀 Quick Start

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
1. Click **"📂 Browse Folder"** — Opens native macOS Finder picker
2. Select your photo directory → Auto-fills in sidebar
3. Click **"🚀 Analyze Photos"** — CLIP + OpenCV scoring begins
4. **Review** the AI-powered gallery with insights
5. **Approve/Reject** photos and export

**Option 2: Command-Line Batch Analysis**
```bash
python analyze_photos.py /path/to/wedding/photos
```
Outputs:
- `photo_analysis.csv` — Detailed metrics + CLIP insights for every photo
- `BEST_PRINTS/` folder — All photos scoring above quality threshold

---

## 📸 Usage Guide

### Via Streamlit Web App

1. **Launch the app:**
   ```bash
   streamlit run app.py
   ```

2. **Select your photo folder:**
   - Click "📂 Browse Folder" for native macOS Finder dialog
   - Or manually paste folder path into text input

3. **Analyze:**
   - Click "🚀 Analyze Photos"
   - Watch real-time progress as CLIP scores each image
   - Perceptual hashing automatically removes burst duplicates

4. **Review summary stats:**
   - Total photos analyzed
   - Highest/average scores
   - Photos filtered for diversity

5. **Browse the AI gallery:**
   - Photos ranked by hybrid score
   - Each card shows: rank, thumbnail, composite score, sharpness, face count, resolution
   - **AI Insight** — "AI says: [matching prompt]" shows what CLIP understood
   - Approve ✓ or Reject ✗ individual photos

6. **Export curated collection:**
   - All approved photos copied to `BEST_PRINTS/`
   - Ready for client review or album creation

### Via Command Line

```bash
python analyze_photos.py /path/to/photos --output-csv results.csv
```

The `BEST_PRINTS` folder will be created with all photos above the quality threshold.

---

## 🛠️ Tech Stack

| Component | Purpose |
|-----------|---------|
| **Python 3.8+** | Core language |
| **CLIP (OpenAI)** | Semantic image understanding, vision-language model |
| **PyTorch** | Deep learning framework for CLIP inference |
| **OpenCV** | Image processing, face detection, Laplacian sharpness |
| **Pillow (PIL)** | Image loading and perceptual hashing |
| **ImageHash** | Perceptual hashing for diversity filtering |
| **NumPy** | Numerical computations and normalization |
| **Streamlit** | Interactive web UI with native folder picker |
| **tqdm** | Progress bars for batch operations |

---

## 💼 Use Cases

- 📷 **Wedding Photography Studios** — Rapidly curate albums from 2000+ photos
- 🎉 **Event Photographers** — Filter high-quality shots from large events
- ✈️ **Travel Bloggers** — Select print-worthy photos from trips
- 📚 **Photo Collections** — Organize and rank personal photo archives
- 🎓 **Photography Students** — Learn computer vision applied to real-world problems

---

## 📊 Sample Output

### CSV Report (`photo_analysis.csv`)

```
rank,filename,final_score,clip_score,clip_prompt,sharpness_raw,exposure_score,resolution_mp,face_count,path
1,IMG_0847.jpg,0.8124,0.89,"a candid emotional wedding moment",298.5,0.92,24.3,2,/photos/IMG_0847.jpg
2,IMG_0891.jpg,0.7856,0.85,"a beautiful well-lit portrait with great composition",287.2,0.88,21.5,1,/photos/IMG_0891.jpg
3,IMG_0742.jpg,0.7642,0.81,"a sharp focused photo with good lighting",275.8,0.85,18.0,0,/photos/IMG_0742.jpg
```

### Web Gallery Display

Each photo card shows:
- **Rank badge** — `#1, #2, #3`
- **Thumbnail** — Preview of the photo
- **Score** — Color-coded composite score (red < 0.5, orange 0.5-0.7, green ≥ 0.7)
- **🤖 AI says** — *"a candid emotional wedding moment"* — CLIP's interpretation
- **Sharpness & Face Count** — Raw metric values
- **Resolution** — Megapixels
- **Approve ✓ / Reject ✗** — Curation controls

### Directory Structure After Export

```
wedding-photos/
├── IMG_0001.jpg
├── IMG_0015.jpg
├── ... (all original photos)
└── BEST_PRINTS/
    ├── IMG_0847.jpg          ← All approved, diverse moments
    ├── IMG_0891.jpg             (no burst duplicates)
    ├── IMG_0742.jpg
    └── ... (40+ more unique shots)
```

---

## 🔧 Configuration & Customization

### Adjust Scoring Weights

Edit the `SCORE_WEIGHTS` dictionary in `app.py` or `analyze_photos.py`:

```python
SCORE_WEIGHTS_HYBRID = {
    "clip":       0.50,   # AI semantic understanding (most powerful)
    "sharpness":  0.20,   # Laplacian edge clarity
    "exposure":   0.15,   # Histogram-based brightness
    "resolution": 0.10,   # Megapixels
    "faces":      0.05,   # Face detection bonus
}
```

### Modify CLIP Prompts

Customize semantic scoring by editing prompts in `app.py`/`analyze_photos.py`:

```python
CLIP_PROMPTS = [
    "a beautiful well-lit portrait with great composition",
    "a candid emotional wedding moment",
    # Add your own...
]

CLIP_NEGATIVE_PROMPTS = [
    "a blurry dark or poorly composed photo",
    # Add your own...
]
```

### Adjust Similarity Threshold

Change perceptual hashing threshold (0-100, higher = stricter duplicate removal):

```python
filter_similar_photos(results, similarity_threshold=85.0)  # More strict
```

### Threshold-Based Selection

Modify the quality threshold for BEST_PRINTS export:

```python
CLIP_SCORE_THRESHOLD = 0.25  # Lower = include more photos
```

---

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

- [ ] Additional metrics (color balance, composition, motion blur detection)
- [ ] Batch processing improvements and parallelization
- [ ] GUI file browser integration
- [ ] Cloud storage support (AWS S3, Google Drive)
- [ ] Advanced filtering by face count, date range, etc.
- [ ] Dark/light theme toggle in Streamlit UI
- [ ] Comparison view showing before/after edits

**To contribute:**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 📧 Questions & Support

For issues, feature requests, or questions:
- Open an [Issue](https://github.com/yourusername/wedding-photo-curator/issues)
- Check [Discussions](https://github.com/yourusername/wedding-photo-curator/discussions)

---

## 🎓 Technical Highlights

This project demonstrates advanced computer vision and AI:

- **Vision-Language Model (CLIP)** — Semantic understanding of image content using OpenAI's CLIP
- **Hybrid Scoring** — Combines AI semantics with traditional computer vision metrics
- **Perceptual Hashing** — Efficient near-duplicate detection (not pixel-perfect matching)
- **Image Processing** — OpenCV Haar Cascades, Laplacian edge detection, histogram analysis
- **GPU Acceleration** — Automatic CUDA detection for PyTorch; CPU fallback
- **Data Normalization** — Min-max scaling for consistent batch-relative scoring
- **Modern Web Development** — Streamlit with native OS file dialogs, real-time progress
- **Production-Grade Code** — Clean architecture, error handling, logging, type hints
- **Offline & Free** — Runs completely locally; no cloud APIs or fees

---

**Built with ❤️ for photographers and AI enthusiasts**
