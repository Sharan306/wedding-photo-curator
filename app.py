"""
Streamlit web application for quality-first wedding photo curation.
Displays curated photos that passed hard rejections and diversity filtering.
"""

import json
import sys
import shutil
import subprocess
from pathlib import Path

import streamlit as st
from PIL import Image

# Configuration
BEST_PRINTS_DIR = "BEST_PRINTS"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
CACHE_FILE = "photo_analysis_cache.json"
CURATION_FILE = "photo_curation.csv"

# Streamlit page configuration
st.set_page_config(
    page_title="Wedding Photo Curator",
    page_icon="💍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .photo-gallery {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
        gap: 20px;
        margin-top: 20px;
    }
    .photo-card {
        background-color: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .photo-card:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .photo-card img {
        width: 100%;
        height: 250px;
        object-fit: cover;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def scan_images(photo_dir: Path) -> list[Path]:
    """Recursively find all supported images."""
    return sorted(
        p
        for p in photo_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
        and BEST_PRINTS_DIR not in p.parts
    )


def load_curation_results(curation_csv: Path) -> list[dict]:
    """Load curation results from CSV."""
    import csv

    if not curation_csv.exists():
        return []

    try:
        with open(curation_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception:
        return []


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "folder_path" not in st.session_state:
        st.session_state.folder_path = ""
    if "curation_results" not in st.session_state:
        st.session_state.curation_results = []
    if "is_analyzing" not in st.session_state:
        st.session_state.is_analyzing = False


def main():
    """Main Streamlit application."""
    initialize_session_state()

    # Header
    st.title("💍 Wedding Photo Curator")
    st.markdown(
        "**Quality-first photo curation**: Selects only truly great photos using hard rejection rules and diversity filtering."
    )
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")

        # Folder path input
        folder_input = st.text_input(
            "Enter photo folder path:",
            value=st.session_state.folder_path,
            placeholder="/path/to/wedding/photos",
        )

        st.markdown("---")

        # Analyze button
        if st.button("🚀 Analyze & Curate Photos", type="primary", use_container_width=True):
            if not folder_input:
                st.error("Please enter a folder path.")
            else:
                folder_path = Path(folder_input)
                if not folder_path.exists() or not folder_path.is_dir():
                    st.error(f"Folder not found: {folder_path}")
                else:
                    images = scan_images(folder_path)
                    if not images:
                        st.warning("No supported images found in the folder.")
                    else:
                        # Run analysis
                        st.info(f"Found {len(images)} images. Running curation analysis...")

                        try:
                            # Call analyze_photos.py
                            result = subprocess.run(
                                [sys.executable, "analyze_photos.py", str(folder_path)],
                                cwd=Path(__file__).parent,
                                capture_output=True,
                                text=True,
                                timeout=600,
                            )

                            if result.returncode == 0:
                                # Load results
                                curation_csv = folder_path / CURATION_FILE
                                results = load_curation_results(curation_csv)
                                st.session_state.curation_results = results
                                st.session_state.folder_path = folder_input
                                st.success(f"✓ Curation complete! Selected {len(results)} photos.")
                                st.rerun()
                            else:
                                st.error(f"Analysis failed: {result.stderr}")
                        except subprocess.TimeoutExpired:
                            st.error("Analysis timed out (over 10 minutes).")
                        except Exception as e:
                            st.error(f"Error during analysis: {e}")

    # Main content
    if st.session_state.curation_results:
        results = st.session_state.curation_results
        total_photos = len(scan_images(Path(st.session_state.folder_path)))

        # Summary
        st.subheader("✨ Curation Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Analyzed", total_photos)
        with col2:
            st.metric("Selected Curated", len(results))
        with col3:
            percentage = (len(results) / total_photos * 100) if total_photos > 0 else 0
            st.metric("Selection Rate", f"{percentage:.1f}%")

        st.markdown(f"**Selected {len(results)} truly great photos from {total_photos} analyzed.**")
        st.markdown("---")

        # Gallery
        st.subheader("🖼️ Curated Photo Gallery")

        cols_per_row = 4
        for i in range(0, len(results), cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx, col in enumerate(cols):
                if i + col_idx >= len(results):
                    break

                result = results[i + col_idx]
                photo_path = Path(result["path"])

                try:
                    with col:
                        # Display image
                        image = Image.open(photo_path)
                        st.image(image, use_column_width=True)

                        # Show rank
                        rank = result.get("rank", "?")
                        st.caption(f"**#{rank}** · {result.get('filename', 'Unknown')}")

                except Exception as e:
                    with col:
                        st.warning(f"Could not load: {e}")

        # Export section
        st.markdown("---")
        st.subheader("💾 Export Selected Photos")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Copy to BEST_PRINTS", use_container_width=True):
                try:
                    folder_path = Path(st.session_state.folder_path)
                    dest = folder_path / BEST_PRINTS_DIR
                    dest.mkdir(exist_ok=True)

                    # Copy all selected photos
                    for result in results:
                        src = Path(result["path"])
                        if src.exists():
                            shutil.copy2(src, dest / src.name)

                    st.success(f"✓ Copied {len(results)} photos to {BEST_PRINTS_DIR}/")
                except Exception as e:
                    st.error(f"Export failed: {e}")

        with col2:
            if st.button("📊 Download CSV Report", use_container_width=True):
                try:
                    folder_path = Path(st.session_state.folder_path)
                    csv_file = folder_path / CURATION_FILE
                    if csv_file.exists():
                        with open(csv_file, "rb") as f:
                            st.download_button(
                                label="Download photo_curation.csv",
                                data=f.read(),
                                file_name="photo_curation.csv",
                                mime="text/csv",
                            )
                except Exception as e:
                    st.error(f"Download failed: {e}")

    else:
        st.info("👈 Enter a folder path and click 'Analyze & Curate Photos' to get started.")


if __name__ == "__main__":
    main()
