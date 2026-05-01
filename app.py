"""Streamlit web application for AI-powered wedding photo curation."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import imagehash
import numpy as np
import streamlit as st
import torch
from PIL import Image
from tkinter import filedialog, Tk

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
BEST_PRINTS_DIR = "BEST_PRINTS"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
CLIP_SCORE_THRESHOLD = 0.25

# CLIP prompts for semantic scoring
CLIP_PROMPTS = [
    "a beautiful well-lit portrait with great composition",
    "a sharp focused photo with good lighting",
    "a candid emotional wedding moment",
    "a beautiful pose with flattering angle",
]

CLIP_NEGATIVE_PROMPTS = [
    "a blurry dark or poorly composed photo",
]

SCORE_WEIGHTS = {
    "clip": 0.50,
    "sharpness": 0.20,
    "exposure": 0.15,
    "resolution": 0.10,
    "faces": 0.05,
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
# CLIP Model (Semantic Scoring)
# ============================================================================

_clip_model = None
_clip_preprocess = None
_clip_device = None


@st.cache_resource
def load_clip_model():
    """Load CLIP model on first use (lazy loading with caching)."""
    if not CLIP_AVAILABLE:
        return None, None, None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


def compute_clip_score(pil_img: Image.Image) -> tuple[float, str]:
    """
    Compute CLIP semantic score for an image.
    Returns (score, best_matching_prompt).
    """
    if not CLIP_AVAILABLE:
        return 0.5, "CLIP unavailable"
    
    try:
        model, preprocess, device = load_clip_model()
        if model is None:
            return 0.5, "CLIP not loaded"
        
        # Encode image
        image_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Encode positive prompts
        pos_texts = [f"This is {p}" for p in CLIP_PROMPTS]
        pos_tokens = clip.tokenize(pos_texts).to(device)
        with torch.no_grad():
            pos_features = model.encode_text(pos_tokens)
            pos_features /= pos_features.norm(dim=-1, keepdim=True)
        
        # Encode negative prompts
        neg_texts = [f"This is {p}" for p in CLIP_NEGATIVE_PROMPTS]
        neg_tokens = clip.tokenize(neg_texts).to(device)
        with torch.no_grad():
            neg_features = model.encode_text(neg_tokens)
            neg_features /= neg_features.norm(dim=-1, keepdim=True)
        
        # Compute similarities
        pos_similarity = image_features @ pos_features.T
        neg_similarity = image_features @ neg_features.T
        
        pos_max = pos_similarity.max().item()
        neg_max = neg_similarity.max().item() if len(CLIP_NEGATIVE_PROMPTS) > 0 else 0.0
        
        # Score: positive - negative, normalized to [0, 1]
        raw_score = pos_max - (0.3 * neg_max)
        score = max(0.0, min(1.0, raw_score))
        
        # Find best matching positive prompt
        best_idx = pos_similarity[0].argmax().item()
        best_prompt = CLIP_PROMPTS[best_idx]
        
        return score, best_prompt
    
    except Exception as exc:
        logger.warning(f"CLIP scoring error: {exc}")
        return 0.5, "CLIP error"


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
    clip_score, clip_prompt = compute_clip_score(pil_img)

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
        "clip_score": clip_score,
        "clip_prompt": clip_prompt,
    }


def _minmax_normalise(records: list[dict], key: str, out_key: str) -> None:
    """Min-max normalise a column across the batch."""
    values = [r[key] for r in records]
    lo, hi = min(values), max(values)
    spread = hi - lo or 1.0
    for r in records:
        r[out_key] = (r[key] - lo) / spread


def compute_final_scores(records: list[dict]) -> list[dict]:
    """Normalises raw metrics, computes weighted hybrid scores (CLIP + OpenCV), and ranks."""
    _minmax_normalise(records, "sharpness_raw", "sharpness_norm")
    _minmax_normalise(records, "contrast_raw", "contrast_norm")

    for r in records:
        r["final_score"] = round(
            SCORE_WEIGHTS["clip"] * r["clip_score"]
            + SCORE_WEIGHTS["sharpness"] * r["sharpness_norm"]
            + SCORE_WEIGHTS["exposure"] * r["exposure_score"]
            + SCORE_WEIGHTS["resolution"] * r["resolution_score"]
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
# Diversity Filtering (Near-Duplicate Removal)
# ============================================================================

def compute_perceptual_hash(path: Path) -> Optional[imagehash.ImageHash]:
    """Compute perceptual hash for an image."""
    try:
        pil_img = Image.open(path).convert("RGB")
        return imagehash.phash(pil_img, hash_size=8)
    except Exception as exc:
        logger.warning(f"Could not hash {path.name}: {exc}")
        return None


def hash_similarity(hash1: imagehash.ImageHash, hash2: imagehash.ImageHash) -> float:
    """
    Compute similarity between two perceptual hashes as a percentage (0-100).
    Hamming distance 0 = identical (100% similar).
    Hamming distance 64 = completely different (0% similar).
    """
    max_distance = 64  # 8x8 hash = 64 bits
    hamming_dist = hash1 - hash2
    similarity = max(0, 100 * (1 - hamming_dist / max_distance))
    return similarity


def filter_similar_photos(records: list[dict], similarity_threshold: float = 90.0) -> list[dict]:
    """
    Remove near-duplicate photos, keeping only the highest-scoring from each group.
    
    Args:
        records: Sorted list of photo records (highest score first)
        similarity_threshold: Minimum similarity % to consider photos as duplicates (0-100)
    
    Returns:
        Filtered list with duplicates removed, maintaining original sort order
    """
    if not records:
        return records

    # Compute hashes for all photos
    hashes = {}
    for r in records:
        path = Path(r["path"])
        hash_obj = compute_perceptual_hash(path)
        if hash_obj:
            hashes[r["path"]] = hash_obj

    # Filter: for each kept photo, remove similar photos with lower scores
    kept = []
    removed = set()

    for record in records:
        if record["path"] in removed:
            continue

        kept.append(record)
        kept_hash = hashes.get(record["path"])
        if not kept_hash:
            continue

        # Compare against all remaining photos
        for other_record in records:
            if other_record["path"] in removed or other_record["path"] == record["path"]:
                continue

            other_hash = hashes.get(other_record["path"])
            if not other_hash:
                continue

            similarity = hash_similarity(kept_hash, other_hash)
            if similarity >= similarity_threshold:
                removed.add(other_record["path"])

    duplicate_count = len(records) - len(kept)
    if duplicate_count > 0:
        logger.info(f"Filtered {duplicate_count} near-duplicate photos")

    return kept


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


def open_folder_picker() -> str:
    """Open native macOS Finder folder picker using tkinter."""
    try:
        root = Tk()
        root.withdraw()  # Hide the tkinter window
        root.attributes('-topmost', True)  # Bring dialog to front
        folder_path = filedialog.askdirectory(
            title="Select Wedding Photos Folder",
            initialdir=st.session_state.folder_path or str(Path.home()),
        )
        root.destroy()
        return folder_path
    except Exception as exc:
        logger.error(f"Folder picker error: {exc}")
        return ""


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
        # Filter near-duplicate photos for diversity
        results = filter_similar_photos(results, similarity_threshold=90.0)
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

    st.subheader("Photo Gallery")
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

                # AI insight
                if CLIP_AVAILABLE:
                    st.caption(f"AI says: {photo_data['clip_prompt']}")

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
    st.title("Wedding Photo Curator")
    st.markdown("AI-powered wedding photo selection and ranking system")
    st.markdown("---")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        folder_input = st.text_input(
            "Enter photo folder path:",
            value=st.session_state.folder_path,
            placeholder="/path/to/wedding/photos",
        )

        if st.button("Browse Folder", use_container_width=True):
            selected_folder = open_folder_picker()
            if selected_folder:
                st.session_state.folder_path = selected_folder
                st.rerun()
            else:
                st.warning("No folder selected.")

        st.markdown("---")

        if st.button("Analyze Photos", type="primary", use_container_width=True):
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
                        st.success("Analysis complete!")

    # Main content
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results

        # Summary statistics
        st.subheader("Analysis Summary")
        display_summary_stats(results)
        st.markdown("---")

        # Photo gallery
        display_photo_gallery(results)
        st.markdown("---")

        # Export section
        st.subheader("Export Approved Photos")
        approved_count = len(st.session_state.approved_photos)
        st.info(f"Currently selected: {approved_count} photos")

        if approved_count > 0:
            if st.button(
                f"Export {approved_count} Approved Photos",
                type="primary",
                use_container_width=True,
            ):
                export_approved_photos(Path(st.session_state.folder_path))
        else:
            st.warning("Select photos to approve before exporting.")
    else:
        st.info("Enter a folder path and click 'Analyze Photos' to get started.")


if __name__ == "__main__":
    main()
