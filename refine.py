import cv2
import imagehash
import shutil
from PIL import Image
from pathlib import Path

INPUT_DIR = Path("/Users/SaiSharan/WeddingPhotos/OUTDOOR/BEST_PRINTS_V2")
TARGET_COUNT = 30

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
    sharpness = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 500, 1.0)
    brightness = 1 - abs(gray.mean() - 127) / 127
    score = sharpness * 0.7 + brightness * 0.3
    score_cache[path] = score
    return score

def get_hash(path):
    if path in hash_cache:
        return hash_cache[path]
    h = imagehash.phash(Image.open(path))
    hash_cache[path] = h
    return h

def run_round(input_dir, output_dir, threshold):
    photos = sorted([p for p in input_dir.iterdir() 
                     if p.suffix.lower() in {'.jpg','.jpeg','.png'}])
    if not photos:
        return 0
    
    scenes = []
    current_scene = [photos[0]]
    prev_hash = get_hash(photos[0])
    
    for photo in photos[1:]:
        curr_hash = get_hash(photo)
        if prev_hash - curr_hash <= threshold:
            current_scene.append(photo)
        else:
            scenes.append(current_scene)
            current_scene = [photo]
        prev_hash = curr_hash
    scenes.append(current_scene)
    
    output_dir.mkdir(exist_ok=True)
    for f in output_dir.iterdir():
        f.unlink()
    
    for scene in scenes:
        best = max(scene, key=score_photo)
        shutil.copy2(best, output_dir / best.name)
    
    return len(scenes)

current_dir = INPUT_DIR
round_num = 1
threshold = 18

while True:
    output_dir = INPUT_DIR.parent / f"BEST_PRINTS_ROUND_{round_num}"
    count = run_round(current_dir, output_dir, threshold)
    print(f"Round {round_num}: {count} photos (threshold {threshold})")
    
    if count <= TARGET_COUNT:
        print(f"\n*** FINAL: {count} photos in {output_dir} ***")
        break
    
    threshold += 4
    current_dir = output_dir
    round_num += 1
    
    if round_num > 8:
        break
