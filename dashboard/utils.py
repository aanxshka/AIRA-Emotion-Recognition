"""
Shared utilities and configuration for AIRA dashboard
"""
import streamlit as st
import os
import json
import pandas as pd
from datetime import datetime, timedelta

# ── PATHS ─────────────────────────────────────────────────────────────────────
CONFIG_PATH = "data/settings.json"
LOG_PATH    = "data/event_log.csv"
DATA_PATH   = "data/emotion_data.csv"

DEFAULT_CONFIG = {
    "refresh_interval": 5,
    "log_frequency":    1,
    "conf_red":         40,
    "conf_amber":       60,
    "recovery_frames":  3,
}

# ── CONFIG HELPERS ────────────────────────────────────────────────────────────
def load_config():
    if os.path.exists(CONFIG_PATH):
        cfg = json.load(open(CONFIG_PATH))
        for k, v in DEFAULT_CONFIG.items():
            cfg.setdefault(k, v)
        return cfg
    return DEFAULT_CONFIG.copy()

def save_config(cfg):
    os.makedirs("data", exist_ok=True)
    json.dump(cfg, open(CONFIG_PATH, "w"), indent=2)

# ── LOG HELPERS ───────────────────────────────────────────────────────────────
def load_log():
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH, parse_dates=['timestamp'])
    return pd.DataFrame(columns=[
        'timestamp','primary_emotion','confidence',
        'happy','sad','fear','angry','disgust','neutral',
        'video_quality','audio_quality','confidence_band'
    ])

def append_to_log(df_new, conf_red, conf_amber, log_frequency):
    os.makedirs("data", exist_ok=True)
    existing = load_log()
    def band(c):
        if c < conf_red:   return "low"
        if c < conf_amber: return "medium"
        return "high"
    rows = []
    for _, row in df_new.iloc[::log_frequency].iterrows():
        ts = str(row['timestamp'])
        if not existing.empty and ts in existing['timestamp'].astype(str).values:
            continue
        conf = float(row['confidence'])
        rows.append({
            'timestamp':       row['timestamp'],
            'primary_emotion': row['primary_emotion'],
            'confidence':      conf,
            'happy':           float(row['happy_score']),
            'sad':             float(row['sad_score']),
            'fear':            float(row['fear_score']),
            'angry':           float(row['angry_score']),
            'disgust':         float(row['disgust_score']),
            'neutral':         float(row['neutral_score']),
            'video_quality':   float(row['video_signal_quality']),
            'audio_quality':   float(row['audio_signal_quality']),
            'confidence_band': band(conf),
        })
    if rows:
        new_df = pd.DataFrame(rows)
        result = pd.concat([existing, new_df], ignore_index=True)
        result.to_csv(LOG_PATH, index=False)

# ── CONFIDENCE HELPERS ────────────────────────────────────────────────────────
def conf_color(c, conf_red=40):
    if c < conf_red:   return "#DC2626"
    if c < 60: return "#D97706"
    return "#111111"

def conf_bar_color(c, conf_red=40):
    if c < conf_red:   return "#DC2626"
    if c < 60: return "#D97706"
    return "#111111"

def conf_warning(c, conf_red=40):
    if c < conf_red:
        return '<div style="margin-top:8px;padding:6px 12px;background:#FEE2E2;border-radius:6px;font-size:12px;color:#DC2626;font-weight:600;">⚠ Low confidence — event logged</div>'
    if c < 60:
        return '<div style="margin-top:8px;padding:6px 12px;background:#FEF3C7;border-radius:6px;font-size:12px;color:#D97706;font-weight:600;">⚠ Confidence degraded</div>'
    return ''

def band_badge(band):
    if band == "low":
        return '<span style="background:#FEE2E2;color:#DC2626;font-size:11px;font-weight:600;padding:2px 8px;border-radius:999px;">Low</span>'
    if band == "medium":
        return '<span style="background:#FEF3C7;color:#D97706;font-size:11px;font-weight:600;padding:2px 8px;border-radius:999px;">Medium</span>'
    return '<span style="background:#DCFCE7;color:#166534;font-size:11px;font-weight:600;padding:2px 8px;border-radius:999px;">High</span>'

# ── DEMO DATA ─────────────────────────────────────────────────────────────────
def load_demo_frames():
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
    return df.to_dict('records')
