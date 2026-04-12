import cv2
import numpy as np
import time

from utils import get_face, preprocess_frame
from models import predict_emotion_hse


FRAME_SKIP = 1  # every frame

# ─────────────────────────────────────────────
# Emotion → valence mapping
# ─────────────────────────────────────────────

HSE_EMOTIONS     = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral", "contempt"]
FERPLUS_EMOTIONS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]

POSITIVE_EMOTIONS = {"happy", "happiness"}
NEUTRAL_EMOTIONS  = {"neutral", "surprise"}
NEGATIVE_EMOTIONS = {"angry", "anger", "sad", "sadness", "fear", "disgust", "contempt"}


def scores_to_valence_probs(emotion_scores, emotion_labels):
    """
    Aggregate per-emotion probability scores into
    (score_positive, score_neutral, score_negative).
    These are used for AUC computation.
    """
    score_positive = 0.0
    score_neutral  = 0.0
    score_negative = 0.0
    for label, score in zip(emotion_labels, emotion_scores):
        label = label.lower()
        if label in POSITIVE_EMOTIONS:
            score_positive += float(score)
        elif label in NEUTRAL_EMOTIONS:
            score_neutral  += float(score)
        elif label in NEGATIVE_EMOTIONS:
            score_negative += float(score)
    return score_positive, score_neutral, score_negative


# ─────────────────────────────────────────────
# HSEmotion baseline pipeline
#
# Every 10th frame sampled.
# Full probability aggregation — averages
# valence scores across all frames.
# Final prediction = highest average valence.
# ─────────────────────────────────────────────

def process_video_hsemotion_baseline(video_path, model):

    cap = cv2.VideoCapture(video_path)

    frame_count    = 0
    frames_used    = 0
    valence_scores = []  # list of (score_pos, score_neu, score_neg)

    total_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        frame     = preprocess_frame(frame)
        face_crop, _ = get_face(frame)
        if face_crop is None:
            continue

        emotion, scores = predict_emotion_hse(model, face_crop)
        sp, sn, sneg    = scores_to_valence_probs(scores, HSE_EMOTIONS)
        valence_scores.append((sp, sn, sneg))
        frames_used += 1

    cap.release()
    latency_ms = round((time.time() - total_start) * 1000, 2)

    if not valence_scores:
        return None, None, None, None, None, None, latency_ms, 0

    avg_pos = float(np.mean([v[0] for v in valence_scores]))
    avg_neu = float(np.mean([v[1] for v in valence_scores]))
    avg_neg = float(np.mean([v[2] for v in valence_scores]))

    valence_map       = {"positive": avg_pos, "neutral": avg_neu, "negative": avg_neg}
    predicted_valence = max(valence_map, key=valence_map.get)

    return (predicted_valence,
            round(max(avg_pos, avg_neu, avg_neg), 4),
            predicted_valence,
            round(avg_pos, 4), round(avg_neu, 4), round(avg_neg, 4),
            latency_ms, frames_used)


# ─────────────────────────────────────────────
# HSEmotion improved pipeline
#
# Every 10th frame sampled.
# Confidence-weighted aggregation — each frame's
# valence scores are weighted by InsightFace
# detection confidence before averaging.
# ─────────────────────────────────────────────

def process_video_hsemotion_improved(video_path, model):

    cap = cv2.VideoCapture(video_path)

    frame_count  = 0
    frames_used  = 0
    weighted_pos = 0.0
    weighted_neu = 0.0
    weighted_neg = 0.0
    total_weight = 0.0

    total_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        frame        = preprocess_frame(frame)
        face_crop, det_score = get_face(frame)
        if face_crop is None:
            continue

        emotion, scores = predict_emotion_hse(model, face_crop)
        sp, sn, sneg    = scores_to_valence_probs(scores, HSE_EMOTIONS)

        weight        = det_score if det_score is not None else 1.0
        weighted_pos += sp   * weight
        weighted_neu += sn   * weight
        weighted_neg += sneg * weight
        total_weight += weight
        frames_used  += 1

    cap.release()
    latency_ms = round((time.time() - total_start) * 1000, 2)

    if frames_used == 0 or total_weight == 0:
        return None, None, None, None, None, None, latency_ms, 0

    avg_pos = weighted_pos / total_weight
    avg_neu = weighted_neu / total_weight
    avg_neg = weighted_neg / total_weight

    valence_map       = {"positive": avg_pos, "neutral": avg_neu, "negative": avg_neg}
    predicted_valence = max(valence_map, key=valence_map.get)

    return (predicted_valence,
            round(max(avg_pos, avg_neu, avg_neg), 4),
            predicted_valence,
            round(avg_pos, 4), round(avg_neu, 4), round(avg_neg, 4),
            latency_ms, frames_used)


# ─────────────────────────────────────────────
# FERPlus baseline pipeline
#
# Every 10th frame sampled.
# Full probability aggregation.
# ─────────────────────────────────────────────

def process_video_ferplus(video_path, model):

    cap = cv2.VideoCapture(video_path)

    frame_count    = 0
    frames_used    = 0
    valence_scores = []

    total_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        frame        = preprocess_frame(frame)
        face_crop, _ = get_face(frame)
        if face_crop is None:
            continue

        try:
            gray       = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            resized    = cv2.resize(gray, (64, 64))
            blob       = cv2.dnn.blobFromImage(resized, scalefactor=1.0, size=(64, 64))
            model.setInput(blob)
            raw_scores = model.forward()[0]
            raw_scores = raw_scores - raw_scores.max()
            exp_scores = np.exp(raw_scores)
            probs      = exp_scores / exp_scores.sum()
            sp, sn, sneg = scores_to_valence_probs(probs, FERPLUS_EMOTIONS)
            valence_scores.append((sp, sn, sneg))
            frames_used += 1
        except Exception:
            continue

    cap.release()
    latency_ms = round((time.time() - total_start) * 1000, 2)

    if not valence_scores:
        return None, None, None, None, None, None, latency_ms, 0

    avg_pos = float(np.mean([v[0] for v in valence_scores]))
    avg_neu = float(np.mean([v[1] for v in valence_scores]))
    avg_neg = float(np.mean([v[2] for v in valence_scores]))

    valence_map       = {"positive": avg_pos, "neutral": avg_neu, "negative": avg_neg}
    predicted_valence = max(valence_map, key=valence_map.get)

    return (predicted_valence,
            round(max(avg_pos, avg_neu, avg_neg), 4),
            predicted_valence,
            round(avg_pos, 4), round(avg_neu, 4), round(avg_neg, 4),
            latency_ms, frames_used)


# ─────────────────────────────────────────────
# FERPlus improved pipeline
#
# Every 10th frame sampled.
# Confidence-weighted aggregation.
# ─────────────────────────────────────────────

def process_video_ferplus_improved(video_path, model):

    cap = cv2.VideoCapture(video_path)

    frame_count  = 0
    frames_used  = 0
    weighted_pos = 0.0
    weighted_neu = 0.0
    weighted_neg = 0.0
    total_weight = 0.0

    total_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        frame        = preprocess_frame(frame)
        face_crop, det_score = get_face(frame)
        if face_crop is None:
            continue

        try:
            gray       = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            resized    = cv2.resize(gray, (64, 64))
            blob       = cv2.dnn.blobFromImage(resized, scalefactor=1.0, size=(64, 64))
            model.setInput(blob)
            raw_scores = model.forward()[0]
            raw_scores = raw_scores - raw_scores.max()
            exp_scores = np.exp(raw_scores)
            probs      = exp_scores / exp_scores.sum()
            sp, sn, sneg = scores_to_valence_probs(probs, FERPLUS_EMOTIONS)
            weight        = det_score if det_score is not None else 1.0
            weighted_pos += sp   * weight
            weighted_neu += sn   * weight
            weighted_neg += sneg * weight
            total_weight += weight
            frames_used  += 1
        except Exception:
            continue

    cap.release()
    latency_ms = round((time.time() - total_start) * 1000, 2)

    if frames_used == 0 or total_weight == 0:
        return None, None, None, None, None, None, latency_ms, 0

    avg_pos = weighted_pos / total_weight
    avg_neu = weighted_neu / total_weight
    avg_neg = weighted_neg / total_weight

    valence_map       = {"positive": avg_pos, "neutral": avg_neu, "negative": avg_neg}
    predicted_valence = max(valence_map, key=valence_map.get)

    return (predicted_valence,
            round(max(avg_pos, avg_neu, avg_neg), 4),
            predicted_valence,
            round(avg_pos, 4), round(avg_neu, 4), round(avg_neg, 4),
            latency_ms, frames_used)


# ─────────────────────────────────────────────
# MediaPipe baseline pipeline
#
# Geometry-based — no trained classifier.
# Derives soft valence probabilities from
# facial landmark geometry.
# ─────────────────────────────────────────────

def process_video_mediapipe(video_path):

    import mediapipe as mp

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh    = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)

    frame_count    = 0
    frames_used    = 0
    valence_scores = []

    total_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w   = frame.shape[:2]
        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            continue

        lm = result.multi_face_landmarks[0].landmark

        def pt(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h])

        try:
            chin             = pt(152)
            forehead         = pt(10)
            face_h           = abs(chin[1] - forehead[1]) + 1e-6
            face_w           = abs(pt(234)[0] - pt(454)[0]) + 1e-6
            upper_lip        = pt(13)
            left_corner      = pt(61)
            right_corner     = pt(291)
            mouth_centre_y   = (left_corner[1] + right_corner[1]) / 2
            mouth_curve_norm = (upper_lip[1] - mouth_centre_y) / face_h
            left_brow_inner  = pt(70)
            right_brow_inner = pt(300)
            brow_ratio       = abs(right_brow_inner[0] - left_brow_inner[0]) / face_w

            raw_pos  = max(0.0, mouth_curve_norm * 10)
            raw_neg  = max(0.0, (0.35 - brow_ratio) * 10)
            raw_neu  = 1.0
            total    = raw_pos + raw_neg + raw_neu + 1e-6
            sp       = raw_pos / total
            sneg     = raw_neg / total
            sn       = raw_neu / total

            valence_scores.append((sp, sn, sneg))
            frames_used += 1
        except Exception:
            continue

    cap.release()
    face_mesh.close()
    latency_ms = round((time.time() - total_start) * 1000, 2)

    if not valence_scores:
        return None, None, None, None, None, None, latency_ms, 0

    avg_pos = float(np.mean([v[0] for v in valence_scores]))
    avg_neu = float(np.mean([v[1] for v in valence_scores]))
    avg_neg = float(np.mean([v[2] for v in valence_scores]))

    valence_map       = {"positive": avg_pos, "neutral": avg_neu, "negative": avg_neg}
    predicted_valence = max(valence_map, key=valence_map.get)

    return (predicted_valence,
            round(max(avg_pos, avg_neu, avg_neg), 4),
            predicted_valence,
            round(avg_pos, 4), round(avg_neu, 4), round(avg_neg, 4),
            latency_ms, frames_used)


# ─────────────────────────────────────────────
# MediaPipe improved pipeline
#
# Every 10th frame sampled.
# Adds eye aperture + mouth velocity cues.
# Temporal consistency filter (3 frames).
# Confidence-weighted aggregation.
# ─────────────────────────────────────────────

MP_CONSISTENCY_WINDOW = 3

def process_video_mediapipe_improved(video_path):

    import mediapipe as mp

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh    = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)

    frame_count       = 0
    frames_used       = 0
    raw_scores        = []
    consistent_scores = []
    prev_mouth_curve  = None
    consecutive_count = 0
    last_valence      = None

    total_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w   = frame.shape[:2]
        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            prev_mouth_curve  = None
            consecutive_count = 0
            last_valence      = None
            continue

        lm = result.multi_face_landmarks[0].landmark

        def pt(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h])

        try:
            chin             = pt(152)
            forehead         = pt(10)
            face_h           = abs(chin[1] - forehead[1]) + 1e-6
            face_w           = abs(pt(234)[0] - pt(454)[0]) + 1e-6
            upper_lip        = pt(13)
            left_corner      = pt(61)
            right_corner     = pt(291)
            mouth_centre_y   = (left_corner[1] + right_corner[1]) / 2
            mouth_curve_norm = (upper_lip[1] - mouth_centre_y) / face_h
            left_brow_inner  = pt(70)
            right_brow_inner = pt(300)
            brow_ratio       = abs(right_brow_inner[0] - left_brow_inner[0]) / face_w
            lower_inner_lip  = pt(14)
            mouth_open       = abs(lower_inner_lip[1] - upper_lip[1]) / face_h
            left_eye_open    = abs(pt(159)[1] - pt(145)[1]) / face_h
            right_eye_open   = abs(pt(386)[1] - pt(374)[1]) / face_h
            eye_open_avg     = (left_eye_open + right_eye_open) / 2

            if prev_mouth_curve is not None:
                curve_velocity = mouth_curve_norm - prev_mouth_curve
            else:
                curve_velocity = 0.0
            prev_mouth_curve = mouth_curve_norm

            velocity_bonus = max(0.0, curve_velocity * 5)
            raw_pos  = max(0.0, mouth_curve_norm * 10 + velocity_bonus)
            raw_neg  = max(0.0, (0.35 - brow_ratio) * 10)
            raw_neu  = 1.0
            total    = raw_pos + raw_neg + raw_neu + 1e-6
            sp       = raw_pos / total
            sneg     = raw_neg / total
            sn       = raw_neu / total

            valence_map  = {"positive": sp, "neutral": sn, "negative": sneg}
            this_valence = max(valence_map, key=valence_map.get)

            raw_scores.append((sp, sn, sneg))

            if this_valence == last_valence:
                consecutive_count += 1
            else:
                consecutive_count = 1
                last_valence      = this_valence

            if consecutive_count >= MP_CONSISTENCY_WINDOW:
                consistent_scores.append((sp, sn, sneg))

            frames_used += 1

        except Exception:
            prev_mouth_curve  = None
            consecutive_count = 0
            last_valence      = None
            continue

    cap.release()
    face_mesh.close()
    latency_ms = round((time.time() - total_start) * 1000, 2)

    scores_to_use = consistent_scores if consistent_scores else raw_scores

    if not scores_to_use:
        return None, None, None, None, None, None, latency_ms, 0

    avg_pos = float(np.mean([v[0] for v in scores_to_use]))
    avg_neu = float(np.mean([v[1] for v in scores_to_use]))
    avg_neg = float(np.mean([v[2] for v in scores_to_use]))

    valence_map       = {"positive": avg_pos, "neutral": avg_neu, "negative": avg_neg}
    predicted_valence = max(valence_map, key=valence_map.get)

    return (predicted_valence,
            round(max(avg_pos, avg_neu, avg_neg), 4),
            predicted_valence,
            round(avg_pos, 4), round(avg_neu, 4), round(avg_neg, 4),
            latency_ms, frames_used)


# ─────────────────────────────────────────────
# Unified entry point
#
# Returns:
#   predicted_raw_label, predicted_score,
#   predicted_valence,
#   score_positive, score_neutral, score_negative,
#   latency_ms, frames_used
# ─────────────────────────────────────────────

def process_video(video_path, library_type, model):

    if library_type == "hsemotion":
        return process_video_hsemotion_baseline(video_path, model)

    elif library_type == "hsemotion_improved":
        return process_video_hsemotion_improved(video_path, model)

    elif library_type == "ferplus":
        return process_video_ferplus(video_path, model)

    elif library_type == "ferplus_improved":
        return process_video_ferplus_improved(video_path, model)

    elif library_type == "mediapipe":
        return process_video_mediapipe(video_path)

    elif library_type == "mediapipe_improved":
        return process_video_mediapipe_improved(video_path)

    else:
        raise ValueError(f"Unknown library type: {library_type}")