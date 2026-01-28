from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
import pickle
import face_recognition

IMAGES_DIR = Path("data/images")
ENC_DIR = Path("data/encodings")
ENC_FILE = ENC_DIR / "encodings.pkl"

def ensure_dirs():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ENC_DIR.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

def capture_images(user_id: str, num_images: int = 30, cam_index: int = 0):
    """
    Captures num_images face images for a user and saves them.
    Returns (saved_count, last_frame_rgb_for_preview)
    """
    user_folder = IMAGES_DIR / user_id
    user_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try changing camera index.")

    saved = 0
    last_rgb = None

    try:
        while saved < num_images:
            ret, frame = cap.read()
            if not ret:
                continue

            # Resize for speed
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            boxes = face_recognition.face_locations(rgb_small, model="hog")
            # Draw boxes on original frame for feedback
            for (top, right, bottom, left) in boxes:
                top2, right2, bottom2, left2 = [v * 2 for v in (top, right, bottom, left)]
                cv2.rectangle(frame, (left2, top2), (right2, bottom2), (0, 255, 0), 2)

            # Save only if exactly one face is found (clean dataset)
            if len(boxes) == 1:
                # Crop face from original (higher res)
                (top, right, bottom, left) = boxes[0]
                top2, right2, bottom2, left2 = [v * 2 for v in (top, right, bottom, left)]
                face_crop = frame[max(0, top2):max(0, bottom2), max(0, left2):max(0, right2)]

                if face_crop.size > 0:
                    out_path = user_folder / f"{user_id}_{saved+1:03d}.jpg"
                    cv2.imwrite(str(out_path), face_crop)
                    saved += 1

            last_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    finally:
        cap.release()

    return saved, last_rgb

def train_encodings():
    """
    Reads images from data/images/<user_id> and creates face encodings.
    Saves to data/encodings/encodings.pkl
    Returns dict with stats.
    """
    ensure_dirs()
    encodings = []
    labels = []

    user_folders = [p for p in IMAGES_DIR.iterdir() if p.is_dir()]
    for uf in user_folders:
        user_id = uf.name
        for img_path in uf.glob("*.jpg"):
            image = face_recognition.load_image_file(str(img_path))
            boxes = face_recognition.face_locations(image, model="hog")
            if len(boxes) != 1:
                continue
            enc = face_recognition.face_encodings(image, boxes)[0]
            encodings.append(enc)
            labels.append(user_id)

    data = {"encodings": encodings, "labels": labels}
    ENC_DIR.mkdir(parents=True, exist_ok=True)
    with open(ENC_FILE, "wb") as f:
        pickle.dump(data, f)

    return {
        "users_found": len(user_folders),
        "total_images_used": len(encodings),
        "enc_file": str(ENC_FILE)
    }

def load_encodings():
    if not ENC_FILE.exists():
        return None
    with open(ENC_FILE, "rb") as f:
        return pickle.load(f)

def recognize_from_frame(frame_bgr, known_data, tolerance: float = 0.45):
    """
    Returns list of tuples: (user_id or 'Unknown', box)
    box = (top, right, bottom, left) in original frame coords
    """
    # Smaller image for speed
    small = cv2.resize(frame_bgr, (0, 0), fx=0.5, fy=0.5)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb_small, model="hog")
    encs = face_recognition.face_encodings(rgb_small, boxes)

    results = []
    for enc, (top, right, bottom, left) in zip(encs, boxes):
        top2, right2, bottom2, left2 = [v * 2 for v in (top, right, bottom, left)]

        name = "Unknown"
        if known_data and known_data["encodings"]:
            matches = face_recognition.compare_faces(known_data["encodings"], enc, tolerance=tolerance)
            if True in matches:
                # Pick best match by distance
                dists = face_recognition.face_distance(known_data["encodings"], enc)
                best_idx = int(np.argmin(dists))
                if matches[best_idx]:
                    name = known_data["labels"][best_idx]

        results.append((name, (top2, right2, bottom2, left2)))

    return results
