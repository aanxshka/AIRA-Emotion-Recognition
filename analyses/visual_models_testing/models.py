import os
import cv2
import hsemotion_onnx.facial_emotions as _hse_module
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

ONNX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_models")

# ─────────────────────────────────────────────
# Monkey-patch get_model_path to load from disk
# ─────────────────────────────────────────────

_LOCAL_MODELS = {
    'enet_b2_8':           os.path.join(ONNX_DIR, 'enet_b2_8.onnx'),
    'enet_b0_8_best_vgaf': os.path.join(ONNX_DIR, 'enet_b0_8_best_vgaf.onnx'),
    'enet_b0_8_va_mtl':    os.path.join(ONNX_DIR, 'enet_b0_8_va_mtl.onnx'),
}

def _local_get_model_path(model_name):
    if model_name in _LOCAL_MODELS:
        return _LOCAL_MODELS[model_name]
    raise ValueError(f"Model '{model_name}' not found in onnx_models/.")

_hse_module.get_model_path = _local_get_model_path


# ─────────────────────────────────────────────
# HSEmotion loaders
# ─────────────────────────────────────────────

def load_hse_b2():
    return HSEmotionRecognizer(model_name='enet_b2_8')

def load_hse_vgaf():
    return HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf')

def load_hse_va_mtl():
    return HSEmotionRecognizer(model_name='enet_b0_8_va_mtl')


# ─────────────────────────────────────────────
# FERPlus loader
# ─────────────────────────────────────────────

FERPLUS_ONNX = os.path.join(ONNX_DIR, "emotion-ferplus-8.onnx")

def load_ferplus():
    if not os.path.exists(FERPLUS_ONNX):
        raise FileNotFoundError(
            f"FERPlus model not found at {FERPLUS_ONNX}.\n"
            "Download with:\n"
            "  curl -L -o onnx_models/emotion-ferplus-8.onnx "
            "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/"
            "emotion_ferplus/model/emotion-ferplus-8.onnx"
        )
    return cv2.dnn.readNetFromONNX(FERPLUS_ONNX)


# ─────────────────────────────────────────────
# MediaPipe loader
# ─────────────────────────────────────────────

def load_mediapipe():
    return "mediapipe"


# ─────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────

def load_models():
    models = {}

    # ── HSEmotion baseline ────────────────────
    models["HSE_enet_b2"]                 = ("hsemotion",          load_hse_b2())
    models["HSE_enet_b0_vgaf"]            = ("hsemotion",          load_hse_vgaf())
    models["HSE_enet_b0_va_mtl"]          = ("hsemotion",          load_hse_va_mtl())

    # ── HSEmotion improved ────────────────────
    models["HSE_enet_b2_improved"]        = ("hsemotion_improved", load_hse_b2())
    models["HSE_enet_b0_vgaf_improved"]   = ("hsemotion_improved", load_hse_vgaf())
    models["HSE_enet_b0_va_mtl_improved"] = ("hsemotion_improved", load_hse_va_mtl())

    # ── FERPlus baseline + improved ──────────
    models["FERPlus_OpenCV"]              = ("ferplus",            load_ferplus())
    models["FERPlus_OpenCV_improved"]     = ("ferplus_improved",   load_ferplus())

    # ── MediaPipe baseline + improved ────────
    models["MediaPipe_Geometry"]          = ("mediapipe",          load_mediapipe())
    models["MediaPipe_Geometry_improved"] = ("mediapipe_improved", load_mediapipe())

    return models


# ─────────────────────────────────────────────
# HSEmotion inference helper
# ─────────────────────────────────────────────

def predict_emotion_hse(model, face_img):
    """face_img: BGR numpy array (pre-cropped face)"""
    emotion, scores = model.predict_emotions(face_img, logits=False)
    return emotion, scores