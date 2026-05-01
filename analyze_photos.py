#!/usr/bin/env python3
"""
Wedding Photo Analyzer
======================
Scores and ranks a folder of wedding photos using computer vision metrics:
sharpness, exposure quality, resolution, contrast, and face detection.

Outputs a ranked CSV report and copies the top N photos to a BEST_PRINTS folder.
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Optional

import cv2
import imagehash
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

DEFAULT_TOP_N = 50
CLIP_SCORE_THRESHOLD = 0.25  # Select photos above this CLIP score

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

# Weighted contribution of each metric to the final composite score.
# CLIP score now carries 50% weight with combined OpenCV metrics at 50%
SCORE_WEIGHTS_HYBRID = {
    "clip":       0.50,   # AI semantic understanding
    "sharpness":  0.20,   # Laplacian edge clarity
    "exposure":   0.15,   # Histogram-based brightness
    "resolution": 0.10,   # Megapixels
    "faces":      0.05,   # Face detection bonus
}


# ---------------------------------------------------------------------------
# CLIP Model (Semantic Scoring)
# ---------------------------------------------------------------------------

_clip_model = None
_clip_preprocess = None
_clip_device = None


def load_clip_model():
    """Load CLIP model on first use (lazy loading)."""
    global _clip_model, _clip_preprocess, _clip_device
    if _clip_model is not None:
        return _clip_model, _clip_preprocess, _clip_device
    
    if not CLIP_AVAILABLE:
        return None, None, None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    _clip_model = model
    _clip_preprocess = preprocess
    _clip_device = device
    return model, preprocess, device


def compute_clip_score(pil_img: Image.Image) -> tuple[float, str]:
    """
    Compute CLIP semantic score for an image.
    Returns (score, best_matching_prompt).
    Score is positive prompts - negative prompts, normalized to [0, 1].
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
        tqdm.write(f"  CLIP scoring error: {exc}")
        return 0.5, "CLIP error"


# ---------------------------------------------------------------------------
# Metric Computation
# ---------------------------------------------------------------------------

def compute_sharpness(gray: np.ndarray) -> float:
    """
    Laplacian variance — measures high-frequency edge detail.
    Higher = sharper. Raw value; normalised across the batch later.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_exposure(gray: np.ndarray) -> float:
    """
    Exposure quality in [0, 1].
    Penalises clipped shadows/highlights and deviation from a mid-tone mean.
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = hist.sum()
    if total == 0:
        return 0.0

    hist_norm = hist / total
    shadow_clip    = hist_norm[:15].sum()    # very dark pixels
    highlight_clip = hist_norm[240:].sum()   # very bright pixels

    # Score based on proximity to ideal mid-tone mean (128)
    brightness_score = 1.0 - abs(gray.mean() - 128.0) / 128.0
    clip_penalty = (shadow_clip + highlight_clip) * 2.0

    return float(max(0.0, brightness_score - clip_penalty))


def compute_resolution(pil_image: Image.Image) -> tuple[float, float]:
    """Returns (megapixels, normalised_score). Score caps at 1.0 for >= 24 MP."""
    w, h = pil_image.size
    mp = (w * h) / 1_000_000
    return round(mp, 2), min(mp / 24.0, 1.0)


def compute_contrast(gray: np.ndarray) -> float:
    """
    Standard deviation of pixel intensities.
    Raw value; normalised across the batch later.
    """
    return float(gray.std())


def compute_face_score(
    gray: np.ndarray,
    cascade: Optional[cv2.CascadeClassifier],
) -> tuple[int, float]:
    """
    Returns (face_count, normalised_score in [0, 1]).
    Each detected face adds 0.25 to the score, capped at 1.0.
    """
    if cascade is None:
        return 0, 0.0
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    count = int(len(faces))
    return count, min(count * 0.25, 1.0)


# ---------------------------------------------------------------------------
# Per-image Analysis
# ---------------------------------------------------------------------------

def analyse_image(
    path: Path,
    cascade: Optional[cv2.CascadeClassifier],
) -> Optional[dict]:
    """
    Loads one image, computes all raw metrics, and returns a result dict.
    Returns None on any loading or processing error.
    """
    try:
        pil_img = Image.open(path).convert("RGB")
        cv_img  = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray    = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    except Exception as exc:
        tqdm.write(f"  Skipping {path.name}: {exc}")
        return None

    mp, res_score    = compute_resolution(pil_img)
    face_count, face_score = compute_face_score(gray, cascade)
    clip_score, clip_prompt = compute_clip_score(pil_img)

    return {
        "path":             str(path),
        "filename":         path.name,
        "sharpness_raw":    compute_sharpness(gray),
        "exposure_score":   compute_exposure(gray),
        "resolution_mp":    mp,
        "resolution_score": res_score,
        "contrast_raw":     compute_contrast(gray),
        "face_count":       face_count,
        "face_score":       face_score,
        "clip_score":       clip_score,
        "clip_prompt":      clip_prompt,
    }


# ---------------------------------------------------------------------------
# Scoring & Ranking
# ---------------------------------------------------------------------------

def _minmax_normalise(records: list[dict], key: str, out_key: str) -> None:
    """Min-max normalise a column across the batch, writing results to out_key."""
    values = [r[key] for r in records]
    lo, hi = min(values), max(values)
    spread = hi - lo or 1.0
    for r in records:
        r[out_key] = (r[key] - lo) / spread


def compute_final_scores(records: list[dict]) -> list[dict]:
    """
    Normalises raw metrics, computes weighted hybrid scores (CLIP + OpenCV),
    sorts descending, and adds a 1-based rank to each record.
    """
    _minmax_normalise(records, "sharpness_raw", "sharpness_norm")
    _minmax_normalise(records, "contrast_raw",  "contrast_norm")

    for r in records:
        r["final_score"] = round(
            SCORE_WEIGHTS_HYBRID["clip"]       * r["clip_score"]
            + SCORE_WEIGHTS_HYBRID["sharpness"]  * r["sharpness_norm"]
            + SCORE_WEIGHTS_HYBRID["exposure"]   * r["exposure_score"]
            + SCORE_WEIGHTS_HYBRID["resolution"] * r["resolution_score"]
            + SCORE_WEIGHTS_HYBRID["faces"]      * r["face_score"],
            4,
        )

    records.sort(key=lambda x: x["final_score"], reverse=True)
    for rank, r in enumerate(records, start=1):
        r["rank"] = rank

    return records


# ---------------------------------------------------------------------------
# Diversity Filtering (Near-Duplicate Removal)
# ---------------------------------------------------------------------------

def compute_perceptual_hash(path: Path) -> Optional[imagehash.ImageHash]:
    """Compute perceptual hash for an image."""
    try:
        pil_img = Image.open(path).convert("RGB")
        return imagehash.phash(pil_img, hash_size=8)
    except Exception as exc:
        tqdm.write(f"  Could not hash {path.name}: {exc}")
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
    print("\nComputing perceptual hashes for diversity filtering...")
    hashes = {}
    for r in tqdm(records, desc="Hashing", unit="file"):
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
        print(f"  Removed {duplicate_count} near-duplicate photos (>{similarity_threshold}% similar)")

    return kept


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "rank", "filename", "final_score",
    "sharpness_raw", "exposure_score", "resolution_mp",
    "contrast_raw", "face_count", "path",
]


def write_csv(records: list[dict], output_path: Path) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"\nRanked report saved  ->  {output_path}")


def print_top_table(records: list[dict], n: int = 10) -> None:
    col = "{:<6} {:<8} {:<7} {:<7} {}"
    print(f"\nTop {min(n, len(records))} Photos")
    print(col.format("Rank", "Score", "Faces", "MP", "Filename"))
    print("-" * 70)
    for r in records[:n]:
        print(col.format(
            r["rank"],
            f"{r['final_score']:.4f}",
            r["face_count"],
            f"{r['resolution_mp']:.1f}",
            r["filename"],
        ))


def copy_best_prints(records: list[dict], photo_dir: Path, n: int = None, threshold: float = None) -> None:
    """
    Copy best photos to BEST_PRINTS folder.
    If threshold is set, use threshold-based selection. Otherwise use top-N.
    """
    dest = photo_dir / "BEST_PRINTS"
    dest.mkdir(exist_ok=True)
    
    if threshold is not None:
        top = [r for r in records if r["final_score"] >= threshold]
        print(f"\nCopying {len(top)} photos above score {threshold}  ->  {dest}")
    else:
        top = records[:n] if n else records[:DEFAULT_TOP_N]
        print(f"\nCopying top {len(top)} photos  ->  {dest}")
    
    for r in tqdm(top, desc="Copying", unit="file"):
        src = Path(r["path"])
        shutil.copy2(src, dest / src.name)
    print(f"Done. {len(top)} best-print photos ready in:\n  {dest.resolve()}")


# ---------------------------------------------------------------------------
# File Discovery
# ---------------------------------------------------------------------------

def scan_images(photo_dir: Path) -> list[Path]:
    """Recursively find all supported images, excluding the BEST_PRINTS folder."""
    return sorted(
        p for p in photo_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
        and "BEST_PRINTS" not in p.parts
    )


def load_face_cascade() -> Optional[cv2.CascadeClassifier]:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        print("Warning: Face cascade unavailable -- face scoring disabled.")
        return None
    return cascade


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="analyze_photos",
        description="AI-powered wedding photo scorer and selector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_photos.py /path/to/wedding/photos
  python analyze_photos.py ./photos --top 30
  python analyze_photos.py ./photos --output results.csv --no-copy
        """,
    )
    parser.add_argument(
        "photo_dir",
        type=Path,
        help="Root folder containing wedding photos (scanned recursively)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_N,
        metavar="N",
        help=f"Number of top photos to export to BEST_PRINTS (default: {DEFAULT_TOP_N})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="CSV output path (default: <photo_dir>/photo_scores.csv)",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Score and rank only -- skip copying to BEST_PRINTS",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.photo_dir.is_dir():
        sys.exit(f"Error: '{args.photo_dir}' is not a valid directory.")

    csv_out = args.output or args.photo_dir / "photo_scores.csv"

    print("=" * 55)
    print("  Wedding Photo Analyzer")
    print("=" * 55)
    print(f"  Folder : {args.photo_dir.resolve()}")
    print(f"  Top N  : {args.top}")
    print(f"  Output : {csv_out}")
    print("=" * 55)

    images = scan_images(args.photo_dir)
    if not images:
        sys.exit("No supported image files found in the given directory.")
    print(f"\nFound {len(images)} images. Analysing...\n")

    cascade = load_face_cascade()
    records: list[dict] = []

    for img_path in tqdm(images, desc="Scoring", unit="photo"):
        result = analyse_image(img_path, cascade)
        if result:
            records.append(result)

    if not records:
        sys.exit("No images could be processed. Check file formats and permissions.")

    print(f"\nRanking {len(records)} photos...")
    records = compute_final_scores(records)

    print(f"\nFiltering duplicate photos...")
    records = filter_similar_photos(records, similarity_threshold=90.0)

    write_csv(records, csv_out)
    print_top_table(records)

    if not args.no_copy:
        # Use threshold-based selection if CLIP is available, else use top-N
        if CLIP_AVAILABLE:
            copy_best_prints(records, args.photo_dir, threshold=CLIP_SCORE_THRESHOLD)
        else:
            copy_best_prints(records, args.photo_dir, n=args.top)


if __name__ == "__main__":
    main()
