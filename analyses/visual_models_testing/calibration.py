import numpy as np
import os
import json

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]


def save_embedding(user_id, emotion, embedding):

    path = f"user_data/{user_id}.json"

    if os.path.exists(path):
        data = json.load(open(path))
    else:
        data = {}

    if emotion not in data:
        data[emotion] = []

    data[emotion].append(embedding.tolist())

    json.dump(data, open(path, "w"))


def load_embeddings(user_id):

    path = f"user_data/{user_id}.json"

    if not os.path.exists(path):
        return None

    return json.load(open(path))


def reset_calibration(user_id):

    path = f"user_data/{user_id}.json"

    if os.path.exists(path):
        os.remove(path)


def cosine_similarity(a, b):

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def adjust_with_calibration(probs, embedding, user_embeddings):

    base_probs = probs.numpy()[0]

    adjustments = []

    for emotion in EMOTIONS:

        if emotion in user_embeddings:

            stored = np.array(user_embeddings[emotion])

            centroid = stored.mean(axis=0)

            sim = cosine_similarity(embedding, centroid)

        else:

            sim = 0

        adjustments.append(sim)

    adjustments = np.array(adjustments)

    if adjustments.sum() != 0:
        adjustments = adjustments / adjustments.sum()

    adjusted = 0.3 * base_probs + 0.7 * adjustments

    return adjusted