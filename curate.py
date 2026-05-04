"""
Wedding Photo Curator - Smart Edition
Uses zone-based sharpness + color analysis + composition + NIMA aesthetic scoring
"""
import cv2
import imagehash
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

PHOTO_DIR = Path("/Users/SaiSharan/WeddingPhotos/OUTDOOR")
OUTPUT_DIR = PHOTO_DIR / "FINAL_ALBUM"
TARGET_COUNT = 30
SCENE_THRESHOLD = 12  # lower = stricter scene grouping

# Load NIMA model for aesthetic scoring
print("Loading NIMA aesthetic model...")
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
import urllib.request
import os

NIMA_WEIGHTS = Path.home() / ".nima_weights.h5"
if not NIMA_WEIGHTS.exists():
    print("Downloading NIMA weights (one-time, ~17MB)...")
    url = "https://github.com/titu1994/neural-image-assessment/releases/download/v0.5/mobilenet_aesthetic_0.07.h5"
    urllib.request.urlretrieve(url, NIMA_WEIGHTS)

base = MobileNet(input_shape=(224, 224, 3), include_top=False, pooling='avg', weights=None)
x = Dropout(0.75)(base.output)
x = Dense(10, activation='softmax')(x)
nima_model = Model(base.input, x)
nima_model.load_weights(str(NIMA_WEIGHTS))
print("NIMA loaded.")

def nima_score(img):
    """Aesthetic score 1-10 from NIMA model."""
    resized = cv2.resize(img, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x = preprocess_input(rgb.astype(np.float32))
    x = np.expand_dims(x, 0)
    scores = nima_model.predict(x, verbose=0)[0]
    mean = sum((i + 1) * s for i, s in enumerate(scores))
    return mean / 10  # normalize to 0-1

def zone_sharpness(gray):
    """Check sharpness across 9 zones. Reward photos with at least one very sharp zone."""
    h, w = gray.shape
    zones = []
    for i in range(3):
        for j in range(3):
            zone = gray[i*h//3:(i+1)*h//3, j*w//3:(j+1)*w//3]
            zones.append(cv2.Laplacian(zone, cv2.CV_64F).var())
    max_zone = max(zones)
    return min(max_zone / 800, 1.0)

def color_richness(img):
    """Vibrant photos score higher."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].mean() / 255
    return saturation

def histogram_balance(gray):
    """Balanced exposure, no blown highlights or crushed shadows."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    blown = hist[245:].sum()
    crushed = hist[:10].sum()
    return 1 - min(blown + crushed, 1.0)

score_cache = {}
hash_cache = {}

def score_photo(path):
    if path in score_cache:
        return score_cache[path]
    img = cv2.imread(str(path))
    if img is None:
        score_cache[path] = 0
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    sharpness = zone_sharpness(gray)
    color = color_richness(img)
    histogram = histogram_balance(gray)
    aesthetic = nima_score(img)
    
    # Aesthetic is the heaviest weight - it captures "would a human like this"
    score = (aesthetic * 0.5) + (sharpness * 0.25) + (color * 0.15) + (histogram * 0.1)
    score_cache[path] = score
    return score

def get_hash(path):
    if path in hash_cache:
        return hash_cache[path]
    h = imagehash.phash(Image.open(path))
    hash_cache[path] = h
    return h

# Find all photos
photos = sorted([p for p in PHOTO_DIR.iterdir() 
                 if p.suffix.lower() in {'.jpg','.jpeg','.png'}])
print(f"\nFound {len(photos)} photos. Scoring with smart analysis...")

# Score all photos with progress bar
for p in tqdm(photos, desc="Scoring"):
    score_photo(p)

print("\nGrouping by scene...")
scenes = []
current_scene = [photos[0]]
prev_hash = get_hash(photos[0])

for photo in photos[1:]:
    curr_hash = get_hash(photo)
    if prev_hash - curr_hash <= SCENE_THRESHOLD:
        current_scene.append(photo)
    else:
        scenes.append(current_scene)
        current_scene = [photo]
    prev_hash = curr_hash
scenes.append(current_scene)
print(f"Found {len(scenes)} distinct scenes")

# Pick best from each scene
selected = [max(scene, key=score_photo) for scene in scenes]

# If still too many, keep top scoring overall
if len(selected) > TARGET_COUNT:
    selected.sort(key=score_photo, reverse=True)
    selected = selected[:TARGET_COUNT]

# Write final folder
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir()
for p in selected:
    shutil.copy2(p, OUTPUT_DIR / p.name)

print(f"\n*** FINAL: {len(selected)} photos in {OUTPUT_DIR} ***")
