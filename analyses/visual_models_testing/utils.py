import cv2
import numpy as np
from insightface.app import FaceAnalysis

face_app = FaceAnalysis()
face_app.prepare(ctx_id=-1)




# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────

def preprocess_frame(frame):
    """
    Apply histogram equalisation per channel for low-light compensation.
    Works on BGR frames.
    """
    channels = cv2.split(frame)
    eq_channels = [cv2.equalizeHist(c) for c in channels]
    return cv2.merge(eq_channels)


def align_face(frame, face):
    """
    Align face using eye landmarks to correct for sharp upward angles.
    Returns the aligned face crop.
    """
    kps = face.kps.astype(int)  # 5 keypoints: left eye, right eye, nose, left mouth, right mouth
    left_eye  = kps[0]
    right_eye = kps[1]

    # Compute angle between eyes
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Centre of eyes
    eye_centre = ((left_eye[0] + right_eye[0]) // 2,
                  (left_eye[1] + right_eye[1]) // 2)

    # Rotation matrix and apply
    M = cv2.getRotationMatrix2D(eye_centre, angle, scale=1.0)
    aligned = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]),
                             flags=cv2.INTER_CUBIC)

    # Crop bounding box from aligned frame
    x1, y1, x2, y2 = face.bbox.astype(int)
    w = x2 - x1
    h = y2 - y1
    x1 = max(0, int(x1 + 0.1 * w))
    x2 = max(0, int(x2 - 0.1 * w))
    y1 = max(0, int(y1 + 0.1 * h))
    y2 = max(0, int(y2 - 0.1 * h))

    return aligned[y1:y2, x1:x2]


def get_face(frame):
    """
    Detect face, apply alignment and return cropped face.
    Returns (face_crop, detection_confidence) or (None, None).
    """
    faces = face_app.get(frame)

    if len(faces) == 0:
        return None, None

    face = faces[0]

    try:
        face_crop = align_face(frame, face)
    except Exception:
        # Fall back to simple crop if alignment fails
        x1, y1, x2, y2 = face.bbox.astype(int)
        w = x2 - x1
        h = y2 - y1
        x1 = int(x1 + 0.1 * w)
        x2 = int(x2 - 0.1 * w)
        y1 = int(y1 + 0.1 * h)
        y2 = int(y2 - 0.1 * h)
        face_crop = frame[y1:y2, x1:x2]

    if face_crop.size == 0:
        return None, None

    det_score = float(face.det_score) if hasattr(face, 'det_score') else 1.0

    return face_crop, det_score


# ─────────────────────────────────────────────
# Filename parsing
# ─────────────────────────────────────────────

def parse_filename(filename):
    """
    Supports two naming formats:
      - Young: <Name>_<EmotionState>.mov / .mp4
      - Old:   <Name>_Old_<EmotionState>.mov / .mp4

    Returns: (person, age_group, ground_truth_valence)
    """
    filename = filename.strip()
    name = filename.rsplit(".", 1)[0]
    parts = name.split("_")

    if len(parts) < 2:
        raise ValueError("Filename not valid")

    parts_lower = [p.lower() for p in parts]

    if "old" in parts_lower:
        age_group = "old"
        old_idx = parts_lower.index("old")
        person = "_".join(parts[:old_idx]).lower()
    else:
        age_group = "young"
        person = "_".join(parts[:-1]).lower()

    emotion_raw = parts[-1].lower()

    if emotion_raw in ("happy", "positive"):
        ground_truth_valence = "positive"
    elif emotion_raw == "neutral":
        ground_truth_valence = "neutral"
    elif emotion_raw in ("upset", "negative", "sad", "angry", "fear", "disgust"):
        ground_truth_valence = "negative"
    else:
        raise ValueError(f"Emotion label not recognized: {emotion_raw}")

    return person, age_group, ground_truth_valence