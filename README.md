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

- 🎯 **Intelligent Quality Scoring** — Evaluates each photo across 5 metrics: sharpness, exposure, resolution, contrast, and face detection
- 🖼️ **Interactive Visual Gallery** — Browse ranked photos with approve/reject controls in a beautiful, responsive UI
- 📊 **Detailed Metrics Display** — See quality scores, megapixels, face counts, and raw metric values for each photo
- 💾 **Smart Export** — Automatically copy your approved photos to a dedicated `BEST_PRINTS` folder
- 📁 **Batch Processing** — Recursively scans and analyzes entire photo directories
- 🚀 **Production-Ready** — Clean, well-commented code with comprehensive error handling
- 🎨 **Web-Based Interface** — Modern Streamlit UI — no command-line skills required

---

## 🧠 How It Works

Each photo is evaluated using a weighted combination of five quality metrics, normalized across your entire collection:

| Metric | Weight | Description |
|--------|--------|-------------|
| **Sharpness** | 35% | Measures edge clarity using Laplacian variance. Sharp photos are print-ready. |
| **Exposure** | 25% | Assesses brightness balance, penalizing clipped shadows and blown highlights. |
| **Resolution** | 15% | Scores megapixels relative to professional print standards (up to 24 MP). |
| **Contrast** | 15% | Measures tonal separation using pixel intensity standard deviation. |
| **Face Detection** | 10% | Detects human faces using Haar Cascade classifiers; rewards portraits & candids. |

**Final Score** = 0.35×(Sharpness) + 0.25×(Exposure) + 0.15×(Resolution) + 0.15×(Contrast) + 0.10×(Faces)

All raw metrics are min-max normalized before weighting, ensuring consistent scoring across different collections.

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
Then enter your photo folder path in the sidebar and click **Analyze Photos**. Browse results in the interactive gallery, approve/reject photos, and export.

**Option 2: Command-Line Batch Analysis**
```bash
python analyze_photos.py /path/to/wedding/photos
```
Outputs:
- `photo_analysis.csv` — Ranked report with all metrics
- `BEST_PRINTS/` folder — Top 50 photos copied automatically

---

## 📸 Usage Guide

### Via Streamlit Web App

1. **Launch the app:**
   ```bash
   streamlit run app.py
   ```

2. **Enter your photo folder path** in the sidebar (absolute path)

3. **Click "Analyze Photos"** — Processing bar shows progress

4. **Review the summary stats:**
   - Total photos analyzed
   - Highest score achieved
   - Average collection quality

5. **Browse the photo gallery:**
   - Photos are ranked by composite score
   - View detailed metrics per photo
   - Click checkboxes to approve or reject

6. **Export approved photos:**
   - All approved photos are copied to `BEST_PRINTS/`
   - Ready for client review or printing

### Via Command Line

```bash
python analyze_photos.py /path/to/photos --output-csv results.csv
```

The `BEST_PRINTS` folder will be created in your photo directory with the top 50 images.

---

## 🛠️ Tech Stack

| Component | Purpose |
|-----------|---------|
| **Python 3.8+** | Core language |
| **OpenCV** | Image processing, face detection, Laplacian sharpness |
| **Pillow (PIL)** | Image loading and manipulation |
| **NumPy** | Numerical computations and normalization |
| **Streamlit** | Interactive web UI |
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
rank,filename,final_score,sharpness_raw,exposure_score,resolution_mp,contrast_raw,face_count,path
1,photo_001.jpg,0.8432,285.42,0.92,24.3,68.5,2,/path/to/photo_001.jpg
2,photo_015.jpg,0.8201,278.15,0.89,18.5,72.1,1,/path/to/photo_015.jpg
3,photo_042.jpg,0.7956,265.30,0.85,21.0,65.3,0,/path/to/photo_042.jpg
```

### Directory Structure After Export

```
wedding-photos/
├── photo_001.jpg
├── photo_015.jpg
├── photo_042.jpg
├── ... (1997 more original photos)
└── BEST_PRINTS/
    ├── photo_001.jpg          ← Top 50 curated photos
    ├── photo_015.jpg
    └── photo_042.jpg
```

---

## 🔧 Configuration & Customization

### Adjust Scoring Weights

Edit the `SCORE_WEIGHTS` dictionary in `app.py` or `analyze_photos.py`:

```python
SCORE_WEIGHTS = {
    "sharpness":  0.35,   # Increase for studio work
    "exposure":   0.25,   # Increase for outdoor shoots
    "resolution": 0.15,   # Higher for print-focused workflows
    "contrast":   0.15,
    "faces":      0.10,   # Increase for portrait-heavy collections
}
```

### Change Top-K Export Count

```python
TOP_K = 50  # Change to 100, 25, etc.
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

This project demonstrates:

- **Computer Vision**: Image analysis using OpenCV Haar Cascades and Laplacian edge detection
- **Data Normalization**: Min-max scaling across batch datasets
- **Web Development**: Modern Python web UI with Streamlit
- **Software Engineering**: Clean architecture, error handling, logging, type hints
- **Batch Processing**: Efficient recursive file discovery and parallel-safe operations
- **User Experience**: Intuitive UI design with real-time feedback and progress tracking

---

**Built with ❤️ for photographers and AI enthusiasts**
