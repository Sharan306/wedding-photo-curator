"""Streamlit web application for AI-powered wedding photo curation."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
BEST_PRINTS_DIR = "BEST_PRINTS"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

SCORE_WEIGHTS = {
    "sharpness": 0.35,
    "exposure": 0.25,
    "resolution": 0.15,
    "contrast": 0.15,
    "faces": 0.10,
}

# Streamlit page configuration
st.set_page_config(
    page_title="Wedding Photo Curator",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for enhanced styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .photo-card {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
    }
    .rank-badge {
        background-color: #0d47a1;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .score-high {
        color: #2e7d32;
        font-weight: bold;
    }
    .score-medium {
        color: #f57f17;
        font-weight: bold;
    }
    .score-low {
        color: #c62828;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Scoring Functions
# ============================================================================

def compute_sharpness(gray: np.ndarray) -> float:
    """Laplacian variance — measures edge sharpness."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_exposure(gray: np.ndarray) -> float:
    """Exposure quality score in [0, 1]."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = hist.sum()
    if total == 0:
        return 0.0

    hist_norm = hist / total
    shadow_clip = hist_norm[:15].sum()
    highlight_clip = hist_norm[240:].sum()

    brightness_score = 1.0 - abs(gray.mean() - 128.0) / 128.0
    clip_penalty = (shadow_clip + highlight_clip) * 2.0

    return float(max(0.0, brightness_score - clip_penalty))


def compute_resolution(pil_image: Image.Image) -> tuple[float, float]:
    """Returns (megapixels, normalised_score)."""
    w, h = pil_image.size
    mp = (w * h) / 1_000_000
    return round(mp, 2), min(mp / 24.0, 1.0)


def compute_contrast(gray: np.ndarray) -> float:
    """Standard deviation of pixel intensities."""
    return float(gray.std())


def compute_face_score(gray: np.ndarray) -> tuple[int, float]:
    """Returns (face_count, normalised_score in [0, 1])."""
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            return 0, 0.0
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        count = int(len(faces))
        return count, min(count * 0.25, 1.0)
    except Exception:
        return 0, 0.0


def analyse_image(path: Path) -> Optional[Dict]:
    """Load image and compute all raw metrics."""
    try:
        pil_img = Image.open(path).convert("RGB")
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    except Exception as exc:
        logger.warning(f"Skipping {path.name}: {exc}")
        return None

    mp, res_score = compute_resolution(pil_img)
    face_count, face_score = compute_face_score(gray)

    return {
        "path": str(path),
        "filename": path.name,
        "sharpness_raw": compute_sharpness(gray),
        "exposure_score": compute_exposure(gray),
        "resolution_mp": mp,
        "resolution_score": res_score,
        "contrast_raw": compute_contrast(gray),
        "face_count": face_count,
        "face_score": face_score,
    }


def _minmax_normalise(records: list[dict], key: str, out_key: str) -> None:
    """Min-max normalise a column across the batch."""
    values = [r[key] for r in records]
    lo, hi = min(values), max(values)
    spread = hi - lo or 1.0
    for r in records:
        r[out_key] = (r[key] - lo) / spread


def compute_final_scores(records: list[dict]) -> list[dict]:
    """Normalises raw metrics, computes weighted composite scores, and ranks."""
    _minmax_normalise(records, "sharpness_raw", "sharpness_norm")
    _minmax_normalise(records, "contrast_raw", "contrast_norm")

    for r in records:
        r["final_score"] = round(
            SCORE_WEIGHTS["sharpness"] * r["sharpness_norm"]
            + SCORE_WEIGHTS["exposure"] * r["exposure_score"]
            + SCORE_WEIGHTS["resolution"] * r["resolution_score"]
            + SCORE_WEIGHTS["contrast"] * r["contrast_norm"]
            + SCORE_WEIGHTS["faces"] * r["face_score"],
            4,
        )

    records.sort(key=lambda x: x["final_score"], reverse=True)
    for rank, r in enumerate(records, start=1):
        r["rank"] = rank

    return records


def scan_images(photo_dir: Path) -> list[Path]:
    """Recursively find all supported images."""
    return sorted(
        p for p in photo_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
        and "BEST_PRINTS" not in p.parts
    )


# ============================================================================
# Streamlit UI
# ============================================================================


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "approved_photos" not in st.session_state:
        st.session_state.approved_photos = set()
    if "folder_path" not in st.session_state:
        st.session_state.folder_path = ""


def evaluate_image_batch(image_paths: List[Path]) -> List[Dict]:
    """Evaluate a batch of images and return scored results."""
    progress_bar = st.progress(0)
    status_text = st.empty()

    raw_results = []
    for index, image_path in enumerate(image_paths):
        try:
            metrics = analyse_image(image_path)
            if metrics:
                raw_results.append(metrics)
            progress = (index + 1) / len(image_paths)
            progress_bar.progress(progress)
            status_text.text(f"Analyzed {index + 1} / {len(image_paths)} photos")
        except Exception as exc:
            logger.warning("Unable to analyze %s: %s", image_path, exc)

    progress_bar.empty()
    status_text.empty()

    # Compute final scores
    if raw_results:
        results = compute_final_scores(raw_results)
    else:
        results = []

    return results


def get_score_class(score: float) -> str:
    """Return CSS class based on score value."""
    if score >= 0.7:
        return "score-high"
    elif score >= 0.5:
        return "score-medium"
    return "score-low"


def display_summary_stats(results: List[Dict]) -> None:
    """Display summary statistics from the analysis."""
    scores = [r["final_score"] for r in results]
    total_analyzed = len(results)
    top_score = max(scores) if scores else 0
    avg_score = sum(scores) / len(scores) if scores else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Analyzed", total_analyzed)
    with col2:
        st.metric("Top Score", f"{top_score:.4f}")
    with col3:
        st.metric("Average Score", f"{avg_score:.4f}")


def display_photo_gallery(results: List[Dict]) -> None:
    """Display ranked photos in a gallery grid with selection toggles."""
    results_sorted = sorted(results, key=lambda x: x["final_score"], reverse=True)

    st.subheader("📸 Photo Gallery")
    st.markdown("---")

    cols_per_row = 3
    for rank_idx in range(0, len(results_sorted), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx, col in enumerate(cols):
            actual_idx = rank_idx + col_idx
            if actual_idx >= len(results_sorted):
                break

            photo_data = results_sorted[actual_idx]
            photo_id = str(photo_data["path"])

            with col:
                st.markdown(
                    '<div class="photo-card">',
                    unsafe_allow_html=True,
                )

                # Rank badge
                st.markdown(
                    f'<span class="rank-badge">#{actual_idx + 1}</span>',
                    unsafe_allow_html=True,
                )

                # Thumbnail
                try:
                    image = Image.open(photo_data["path"])
                    image.thumbnail((300, 300), Image.Resampling.LANCZOS)
                    st.image(image, use_column_width=True)
                except Exception as exc:
                    st.warning(f"Unable to load thumbnail: {exc}")

                # Metrics
                score_class = get_score_class(photo_data["final_score"])
                st.markdown(
                    f'<p class="{score_class}">Score: {photo_data["final_score"]:.4f}</p>',
                    unsafe_allow_html=True,
                )

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric(
                        "Sharpness",
                        f"{photo_data['sharpness_raw']:.2f}",
                        label_visibility="collapsed",
                    )
                with col_b:
                    st.metric(
                        "Faces",
                        f"{photo_data['face_count']}",
                        label_visibility="collapsed",
                    )

                st.caption(f"📊 Resolution: {photo_data['resolution_mp']:.1f} MP")

                # Approve/Reject buttons
                col_approve, col_reject = st.columns(2)
                with col_approve:
                    if st.checkbox("✓ Approve", key=f"approve_{photo_id}"):
                        st.session_state.approved_photos.add(photo_id)
                    else:
                        st.session_state.approved_photos.discard(photo_id)

                with col_reject:
                    if st.checkbox("✗ Reject", key=f"reject_{photo_id}"):
                        st.session_state.approved_photos.discard(photo_id)

                st.markdown("</div>", unsafe_allow_html=True)


def export_approved_photos(root_path: Path) -> None:
    """Copy approved photos to the BEST_PRINTS folder."""
    destination = root_path / BEST_PRINTS_DIR
    destination.mkdir(parents=True, exist_ok=True)

    export_count = 0
    for photo_id in st.session_state.approved_photos:
        source_path = Path(photo_id)
        if source_path.exists():
            target_path = destination / source_path.name
            if target_path.exists():
                target_path = (
                    destination / f"{source_path.stem}_approved{source_path.suffix}"
                )
            shutil.copy2(source_path, target_path)
            export_count += 1

    st.success(f"✓ Exported {export_count} approved photos to {destination}")
    logger.info("Exported %s approved photos to %s", export_count, destination)


def main() -> None:
    """Main Streamlit application."""
    initialize_session_state()

    # Header
    st.title("💍 Wedding Photo Curator")
    st.markdown("AI-powered wedding photo selection and ranking system")
    st.markdown("---")

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        folder_input = st.text_input(
            "📁 Enter photo folder path:",
            value=st.session_state.folder_path,
            placeholder="/path/to/wedding/photos",
        )

        if st.button("📂 Browse Folder"):
            st.info("Note: On Streamlit Cloud, use the text input to specify your folder path.")

        st.markdown("---")

        if st.button("🚀 Analyze Photos", type="primary", use_container_width=True):
            if not folder_input:
                st.error("Please enter a folder path.")
            else:
                folder_path = Path(folder_input)
                if not folder_path.exists() or not folder_path.is_dir():
                    st.error(f"Folder not found: {folder_path}")
                else:
                    image_paths = scan_images(folder_path)
                    if not image_paths:
                        st.warning("No supported images found in the folder.")
                    else:
                        st.info(f"Found {len(image_paths)} images. Starting analysis...")
                        results = evaluate_image_batch(image_paths)
                        st.session_state.analysis_results = results
                        st.session_state.folder_path = folder_input
                        st.session_state.approved_photos = set()
                        st.success("✓ Analysis complete!")

    # Main content
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results

        # Summary statistics
        st.subheader("📊 Analysis Summary")
        display_summary_stats(results)
        st.markdown("---")

        # Photo gallery
        display_photo_gallery(results)
        st.markdown("---")

        # Export section
        st.subheader("💾 Export Approved Photos")
        approved_count = len(st.session_state.approved_photos)
        st.info(f"Currently selected: {approved_count} photos")

        if approved_count > 0:
            if st.button(
                f"📥 Export {approved_count} Approved Photos",
                type="primary",
                use_container_width=True,
            ):
                export_approved_photos(Path(st.session_state.folder_path))
        else:
            st.warning("Select photos to approve before exporting.")
    else:
        st.info("👈 Enter a folder path and click 'Analyze Photos' to get started.")


if __name__ == "__main__":
    main()
