import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from model_performance import (
    generate_confusion_matrix, accuracy_from_matrix, f1_from_matrix,
    generate_cpu_history, generate_latency, fig_to_b64, EMOTIONS
)

st.set_page_config(
    page_title="AIRA Emotion Detection Dashboard",
    page_icon="🤖",
    layout="wide"
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
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

cfg              = load_config()
REFRESH_INTERVAL = cfg["refresh_interval"]
LOG_FREQUENCY    = cfg["log_frequency"]
CONF_RED         = cfg["conf_red"]
CONF_AMBER       = cfg["conf_amber"]
RECOVERY_FRAMES  = cfg["recovery_frames"]

# ── DEMO NOTES — hardcoded UI logic, not data ─────────────────────────────────
DEMO_NOTES = {
    "😊 Normal Operation":    "The model is detecting a stable Happy emotion with high confidence. Both video and audio feeds are active and signal quality is strong.",
    "🔄 Emotion Cycling":     "The detected emotion is shifting as the subject's mood changes. Each emotion registers across the All Emotion Scores panel in real time.",
    "⚠️ Low Confidence Event": "Confidence has dropped below the threshold. The bar turns red, indicating the model is uncertain about the current emotional state. This event is being captured in the Event Timeline.",
    "📉 Signal Quality Dip":  "Video and audio signal quality is degrading. The Input Diagnostics show a drop in signal strength, which is affecting the model's ability to detect emotion accurately.",
    "📵 Feed Lost":            "The video feed has been lost. Without visual input the model's confidence drops significantly. The feed indicator turns red until the connection is restored.",
}

# ── DEMO SEQUENCE — loaded from CSV ─────────────────────────────────────────
def load_demo_frames():
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
    return df.to_dict('records')

DEMO_FRAMES       = load_demo_frames()
TOTAL_DEMO_FRAMES = len(DEMO_FRAMES)

# ── LOG HELPERS ───────────────────────────────────────────────────────────────
def load_log():
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH, parse_dates=['timestamp'])
    return pd.DataFrame(columns=[
        'timestamp','primary_emotion','confidence',
        'happy','sad','fear','angry','disgust','neutral','surprise',
        'video_quality','audio_quality','confidence_band'
    ])

def append_to_log(df_new):
    os.makedirs("data", exist_ok=True)
    existing = load_log()
    def band(c):
        if c < CONF_RED:   return "low"
        if c < CONF_AMBER: return "medium"
        return "high"
    rows = []
    for _, row in df_new.iloc[::LOG_FREQUENCY].iterrows():
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
            'surprise':        float(row.get('surprise_score', 0)),
            'video_quality':   float(row['video_signal_quality']),
            'audio_quality':   float(row['audio_signal_quality']),
            'confidence_band': band(conf),
        })
    if rows:
        out = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        out.to_csv(LOG_PATH, index=False)

# ── DEMO DATA BUILDER ─────────────────────────────────────────────────────────
def get_demo_data():
    """
    Returns a DataFrame of the last 24 frames leading up to the current frame.
    Timestamps are remapped to NOW so the chart always looks live.
    Frame counter advances ONCE after data is fully built.
    """
    if 'demo_frame' not in st.session_state:
        st.session_state.demo_frame = 0

    current_idx = st.session_state.demo_frame
    history_len = 24
    indices     = [(current_idx - history_len + 1 + i) % TOTAL_DEMO_FRAMES
                   for i in range(history_len)]

    now  = datetime.now()
    rows = []
    for offset, idx in enumerate(indices):
        frame = DEMO_FRAMES[idx].copy()
        frame['timestamp'] = now - timedelta(seconds=(history_len - 1 - offset) * REFRESH_INTERVAL)
        rows.append(frame)

    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Advance frame ONCE after data is fully built
    next_frame = current_idx + 1
    if next_frame >= TOTAL_DEMO_FRAMES:
        st.session_state.demo_complete = True
    else:
        st.session_state.demo_frame = next_frame

    return df

def load_real_data():
    # Kept for reference — no longer used in the app
    pass

# ── CONFIDENCE HELPERS ────────────────────────────────────────────────────────
def conf_color(c):
    if c < CONF_RED:   return "#DC2626"
    if c < CONF_AMBER: return "#D97706"
    return "#111111"

def conf_bar_color(c):
    if c < CONF_RED:   return "#DC2626"
    if c < CONF_AMBER: return "#D97706"
    return "#111111"

def conf_warning(c):
    if c < CONF_RED:
        return '<div style="margin-top:8px;padding:6px 12px;background:#FEE2E2;border-radius:6px;font-size:12px;color:#DC2626;font-weight:600;">⚠ Low confidence — event logged</div>'
    if c < CONF_AMBER:
        return '<div style="margin-top:8px;padding:6px 12px;background:#FEF3C7;border-radius:6px;font-size:12px;color:#D97706;font-weight:600;">⚠ Confidence degraded</div>'
    return ''

def band_badge(band):
    if band == "low":
        return '<span style="background:#FEE2E2;color:#DC2626;font-size:11px;font-weight:600;padding:2px 8px;border-radius:999px;">Low</span>'
    if band == "medium":
        return '<span style="background:#FEF3C7;color:#D97706;font-size:11px;font-weight:600;padding:2px 8px;border-radius:999px;">Medium</span>'
    return '<span style="background:#DCFCE7;color:#166534;font-size:11px;font-weight:600;padding:2px 8px;border-radius:999px;">High</span>'

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');

    /* ── RESET — white background everywhere ── */
    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMainBlockContainer"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stBottom"],
    .block-container, .main, .main > div {
        background-color: #FFFFFF !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .stApp > header { background-color: #FFFFFF !important; border-bottom: none !important; }
    .block-container { padding-top: 4rem; padding-bottom: 2rem; }

    /* ── SIDEBAR — white ── */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] section {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E4E4E7 !important;
    }
    .stRadio label { font-size: 13px !important; color: #374151 !important; }
    [data-testid="stSidebar"] hr { border-color: #E4E4E7 !important; }
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    button[kind="header"] { display: none !important; }

    /* ── WHITE CARDS — st.container(border=True) ── */
    [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #E4E4E7 !important;
        border-radius: 12px !important;
        padding: 20px 24px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
        margin-bottom: 4px !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {
        background-color: #FFFFFF !important;
    }

    /* ── BUTTONS ── */
    .stButton > button {
        background: #FFFFFF !important;
        color: #111111 !important;
        border: 1px solid #E4E4E7 !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .stButton > button:hover { background: #F4F4F5 !important; border-color: #A1A1AA !important; }
    .stButton > button[kind="primary"] { background: #111111 !important; color: #FFFFFF !important; border-color: #111111 !important; }
    .stButton > button[kind="primary"]:hover { background: #374151 !important; }

    /* ── SLIDERS — force black, kill red ── */
    [data-testid="stSlider"] { padding: 0 !important; }
    [data-testid="stSlider"] label { font-size: 13px !important; font-weight: 600 !important; color: #374151 !important; }
    [data-testid="stSlider"] [role="progressbar"] { background-color: #111111 !important; }
    [data-testid="stSlider"] [role="slider"] { background-color: #111111 !important; border-color: #111111 !important; }
    [data-testid="stThumbValue"] { color: #111111 !important; font-weight: 700 !important; }

    /* ── SELECTBOX ── */
    [data-testid="stSelectbox"] label { font-size: 13px !important; font-weight: 600 !important; color: #374151 !important; }
    [data-baseweb="select"] > div { border-color: #E4E4E7 !important; border-radius: 8px !important; }

    /* ── CHART WRAPPER ── */
    .diagnostics-chart-wrapper {
        background-color: #FFFFFF;
        border-radius: 0 0 12px 12px;
        padding: 0 24px 24px 24px;
        margin-top: -24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }
    .diagnostics-chart-wrapper .element-container { margin: 0 !important; padding: 0 !important; }

    /* ── ALERTS ── */
    [data-testid="stAlert"] { border-radius: 8px !important; }
    </style>
""", unsafe_allow_html=True)


# ── SESSION STATE DEFAULTS ────────────────────────────────────────────────────
if 'live_mode' not in st.session_state:
    st.session_state.live_mode = False
if 'session_start' not in st.session_state:
    st.session_state.session_start = datetime.now()
if 'demo_frame' not in st.session_state:
    st.session_state.demo_frame = 0
if 'demo_complete' not in st.session_state:
    st.session_state.demo_complete = False
if 'conf_matrix' not in st.session_state:
    st.session_state.conf_matrix = generate_confusion_matrix()
if 'cpu_history' not in st.session_state:
    st.session_state.cpu_history = generate_cpu_history()
if 'latency' not in st.session_state:
    st.session_state.latency = generate_latency()

# ── NAVIGATION ────────────────────────────────────────────────────────────────
page = st.sidebar.radio(
    "Navigation",
    ["📊 Live Dashboard", "📋 Event Timeline", "🧠 Model Performance", "⚙️ Settings"],
    label_visibility="collapsed"
)
st.sidebar.markdown(
    '<div style="padding:16px 0 8px 0;">' +
    '<div style="font-size:10px;font-weight:700;letter-spacing:0.12em;color:#52525B;text-transform:uppercase;margin-bottom:12px;">Navigation</div>' +
    '</div>',
    unsafe_allow_html=True
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f'<div style="font-size:11px;color:#71717A;line-height:1.8;">' +
    f'<span style="color:#52525B;font-weight:600;">AIRA v1.0</span><br>' +
    f'Refresh: {REFRESH_INTERVAL}s · Log every {LOG_FREQUENCY} frame(s)<br>' +
    f'Red &lt;{CONF_RED}% · Amber &lt;{CONF_AMBER}%' +
    f'</div>',
    unsafe_allow_html=True
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — LIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Live Dashboard":

    header_col, controls_col = st.columns([4, 4])

    with header_col:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:12px;padding-top:4px;">' +
            '<span style="font-size:28px;line-height:1;">🤖</span>' +
            '<div>' +
            '<div style="font-size:11px;font-weight:600;color:#A1A1AA;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:2px;">AIRA · Emotion Detection</div>' +
            '<div style="font-size:22px;font-weight:700;color:#18181B;line-height:1.1;">Developer Dashboard</div>' +
            '</div></div>',
            unsafe_allow_html=True
        )

    with controls_col:
        st.markdown('<div style="padding-top:6px;">', unsafe_allow_html=True)
        btn_col, restart_col, skip_col, prog_col = st.columns([0.8, 0.8, 0.8, 4])

        with btn_col:
            is_live = st.session_state.live_mode
            label   = "⏸ Pause" if is_live else "▶ Play"
            if st.button(label, use_container_width=True):
                if not is_live:
                    st.session_state.demo_frame    = 0
                    st.session_state.session_start = datetime.now()
                    st.session_state.demo_complete = False
                st.session_state.live_mode = not is_live
                st.rerun()

        with restart_col:
            if st.button("↺ Reset", use_container_width=True):
                st.session_state.demo_frame    = 0
                st.session_state.session_start = datetime.now()
                st.session_state.demo_complete = False
                st.session_state.live_mode     = True
                st.rerun()

        with skip_col:
            if st.button("⏭ Skip", use_container_width=True):
                current = st.session_state.demo_frame
                current_scenario = DEMO_FRAMES[current % TOTAL_DEMO_FRAMES]["scenario"]
                # Advance until we hit a frame with a different scenario
                next_frame = current + 1
                while next_frame < TOTAL_DEMO_FRAMES:
                    if DEMO_FRAMES[next_frame]["scenario"] != current_scenario:
                        break
                    next_frame += 1
                if next_frame >= TOTAL_DEMO_FRAMES:
                    st.session_state.demo_complete = True
                else:
                    st.session_state.demo_frame    = next_frame
                    st.session_state.demo_complete = False
                st.session_state.live_mode = True
                st.rerun()

        with prog_col:
            st.markdown('<div id="demo-progress-placeholder" style="padding-top:4px;"></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    @st.fragment(run_every=REFRESH_INTERVAL if st.session_state.live_mode else None)
    def render_dashboard():
        live = st.session_state.live_mode

        try:
            df = get_demo_data()
        except Exception as e:
            st.error(f"Could not load data: {e}")
            return

        if st.session_state.live_mode:
            append_to_log(df.iloc[[-1]])

        latest     = df.iloc[-1]
        emotion    = latest['primary_emotion']
        # Use the normalised emotion score as confidence so left and right panels always match
        emotion_col_map = {
            'Happy': 'happy_score', 'Sad': 'sad_score', 'Fear': 'fear_score',
            'Angry': 'angry_score', 'Disgust': 'disgust_score', 'Neutral': 'neutral_score',
            'Surprise': 'surprise_score'
        }
        confidence = float(latest[emotion_col_map.get(emotion, 'confidence')])
        v_active   = latest['video_feed_active']
        a_active   = latest['audio_feed_active']

        emotions       = ['happy_score','sad_score','fear_score','angry_score','disgust_score','neutral_score','surprise_score']
        emotion_labels = ['Happy','Sad','Fear','Angry','Disgust','Neutral','Surprise']
        emotion_values = [float(latest[e]) for e in emotions]

        vq = float(latest['video_signal_quality'])
        aq = float(latest['audio_signal_quality'])

        if live:
            v_status = "⬤ Active"   if v_active else "⬤ Feed Lost"
            a_status = "⬤ Active"   if a_active else "⬤ Feed Lost"
            v_color  = "#16A34A"    if v_active else "#DC2626"
            a_color  = "#16A34A"    if a_active else "#DC2626"
        else:
            v_status = "⬤ Offline"
            a_status = "⬤ Offline"
            v_color  = "#9CA3AF"
            a_color  = "#9CA3AF"

        # ── PROGRESS BAR ─────────────────────────────────────────────────────
        current_frame = st.session_state.demo_frame
        progress_pct  = current_frame / TOTAL_DEMO_FRAMES
        if current_frame < TOTAL_DEMO_FRAMES:
            current_scenario = DEMO_FRAMES[max(0, current_frame - 1)]["scenario"]
        else:
            current_scenario = "Complete"
        st.markdown(
            '<div style="background:#FFFFFF;border-radius:10px;padding:12px 16px;'
            'box-shadow:0 1px 4px rgba(0,0,0,0.08);margin-bottom:16px;">'
            '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
            f'<span style="font-size:13px;font-weight:600;color:#111111;">Demo Progress</span>'
            f'<span style="font-size:12px;color:#6B7280;">{current_scenario} &nbsp;·&nbsp; {current_frame}/{TOTAL_DEMO_FRAMES} frames</span>'
            '</div>'
            '<div style="background:#E5E7EB;border-radius:999px;height:8px;overflow:hidden;">'
            f'<div style="background:#111111;height:8px;border-radius:999px;width:{progress_pct*100:.1f}%;"></div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        # ── DEMO COMPLETE BANNER ──────────────────────────────────────────────
        if live and st.session_state.get('demo_complete', False):
            st.markdown(
                '<div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:10px;'
                'padding:16px 20px;margin-bottom:16px;display:flex;align-items:center;justify-content:space-between;">'
                '<div>'
                '<div style="font-size:14px;font-weight:700;color:#166534;margin-bottom:4px;">✅ Demo sequence complete</div>'
                '<div style="font-size:13px;color:#166534;">All scenarios have played through. Would you like to run it again?</div>'
                '</div>',
                unsafe_allow_html=True
            )
            if st.button("↺ Run Again", type="primary"):
                st.session_state.demo_frame    = 0
                st.session_state.session_start = datetime.now()
                st.session_state.demo_complete = False
                st.session_state.live_mode     = True
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # ── DEMO SCENARIO BANNER ──────────────────────────────────────────────
        if not st.session_state.get('demo_complete', False) and 'scenario' in latest and pd.notna(latest.get('scenario','')):
            scenario  = latest['scenario']
            demo_note = DEMO_NOTES.get(scenario, '')
            st.markdown(
                f'<div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:10px;'
                f'padding:12px 18px;margin-bottom:16px;">'
                f'<div style="font-size:14px;font-weight:700;color:#1D4ED8;margin-bottom:4px;">'
                f'🎬 Demo: {scenario}</div>'
                f'<div style="font-size:13px;color:#1E40AF;line-height:1.5;">{demo_note}</div>'
                f'</div>',
                unsafe_allow_html=True
            )



        badge = (
            '<span style="background:#DCFCE7;color:#166534;font-size:12px;font-weight:600;padding:3px 10px;border-radius:999px;margin-left:8px;">● LIVE</span>'
            if live else
            '<span style="background:#F4F4F5;color:#6B7280;font-size:12px;font-weight:600;padding:3px 10px;border-radius:999px;margin-left:8px;">⏸ PAUSED</span>'
        )
        st.markdown(
            f'<div style="font-size:12px;color:#9CA3AF;margin-bottom:20px;">Last updated: {datetime.now().strftime("%H:%M:%S")}{badge}</div>',
            unsafe_allow_html=True
        )

        # Emotion rows
        emotion_rows_html = ""
        for label, value in zip(emotion_labels, emotion_values):
            emotion_rows_html += (
                '<div style="margin-bottom:14px;">'
                    '<div style="display:flex;justify-content:space-between;margin-bottom:5px;">'
                        f'<div style="font-weight:600;font-size:14px;color:#111111;">{label}</div>'
                        f'<div style="font-size:13px;color:#6B7280;">{value:.1f}%</div>'
                    '</div>'
                    '<div style="background-color:#E5E7EB;height:6px;border-radius:999px;overflow:hidden;">'
                        f'<div style="background-color:#111111;height:6px;border-radius:999px;width:{min(value,100)}%;"></div>'
                    '</div>'
                '</div>'
            )

        st.markdown((
            '<div style="background-color:#FFFFFF;border-radius:12px;padding:24px;box-shadow:0 1px 4px rgba(0,0,0,0.08);margin-bottom:24px;">'
                '<div style="font-size:18px;font-weight:700;color:#111111;margin-bottom:20px;">Live Monitoring</div>'
                '<div style="display:flex;gap:24px;align-items:flex-start;">'
                    '<div style="flex:0 0 55%;min-width:0;display:flex;flex-direction:column;gap:12px;">'
                        '<div style="background-color:#F4F4F5;border-radius:10px;padding:16px;">'
                            '<div style="font-size:13px;color:#9CA3AF;margin-bottom:8px;">⬤ Current Emotion</div>'
                            f'<div style="font-size:48px;font-weight:700;color:#111111;line-height:1;margin-bottom:10px;">{emotion}</div>'
                            f'<div style="font-size:14px;color:{conf_color(confidence)};margin-bottom:8px;font-weight:600;">Confidence: {confidence:.1f}%</div>'
                            '<div style="background-color:#E5E7EB;border-radius:999px;height:8px;overflow:hidden;">'
                                f'<div style="background-color:{conf_bar_color(confidence)};border-radius:999px;height:8px;width:{min(confidence,100)}%;"></div>'
                            '</div>'
                            + conf_warning(confidence) +
                        '</div>'
                        '<div style="display:flex;gap:12px;">'
                            '<div style="background-color:#F4F4F5;border-radius:10px;padding:12px 16px;flex:1;">'
                                '<div style="font-size:13px;color:#6B7280;margin-bottom:6px;">📹 Video Feed</div>'
                                f'<div style="font-size:14px;font-weight:600;color:{v_color};">{v_status}</div>'
                            '</div>'
                            '<div style="background-color:#F4F4F5;border-radius:10px;padding:12px 16px;flex:1;">'
                                '<div style="font-size:13px;color:#6B7280;margin-bottom:6px;">🎤 Audio Feed</div>'
                                f'<div style="font-size:14px;font-weight:600;color:{a_color};">{a_status}</div>'
                            '</div>'
                        '</div>'
                    '</div>'
                    '<div style="flex:1;min-width:0;">'
                        '<div style="font-size:14px;font-weight:600;color:#374151;margin-bottom:14px;">All Emotion Scores</div>'
                        + emotion_rows_html +
                    '</div>'
                '</div>'
            '</div>'
        ), unsafe_allow_html=True)

        st.markdown((
            '<div style="background-color:#FFFFFF;border-radius:12px 12px 0 0;padding:24px 24px 16px 24px;box-shadow:0 -1px 4px rgba(0,0,0,0.06);">'
                '<div style="font-size:18px;font-weight:700;color:#111111;margin-bottom:20px;">Input Diagnostics</div>'
                '<div style="display:flex;gap:24px;">'
                    '<div style="flex:1;background-color:#F3F4F6;border-radius:10px;padding:16px;">'
                        '<div style="font-size:13px;color:#6B7280;margin-bottom:8px;">📹 Video Signal</div>'
                        f'<div style="font-size:36px;font-weight:700;color:#111111;margin-bottom:12px;">{vq:.0f}%</div>'
                        '<div style="background-color:#E5E7EB;border-radius:999px;height:6px;overflow:hidden;">'
                            f'<div style="background-color:#111111;border-radius:999px;height:6px;width:{min(vq,100)}%;"></div>'
                        '</div>'
                    '</div>'
                    '<div style="flex:1;background-color:#F3F4F6;border-radius:10px;padding:16px;">'
                        '<div style="font-size:13px;color:#6B7280;margin-bottom:8px;">🎤 Audio Signal</div>'
                        f'<div style="font-size:36px;font-weight:700;color:#111111;margin-bottom:12px;">{aq:.0f}%</div>'
                        '<div style="background-color:#E5E7EB;border-radius:999px;height:6px;overflow:hidden;">'
                            f'<div style="background-color:#111111;border-radius:999px;height:6px;width:{min(aq,100)}%;"></div>'
                        '</div>'
                    '</div>'
                    '<div style="flex:1;"></div>'
                    '<div style="flex:1;"></div>'
                '</div>'
            '</div>'
        ), unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['video_signal_quality'],
            mode='lines', name='Video Signal', line=dict(color='#111111', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['audio_signal_quality'],
            mode='lines', name='Audio Signal', line=dict(color='#9CA3AF', width=2, dash='dot')
        ))
        fig.update_layout(
            title=dict(text="Signal Quality Over Time", font=dict(size=14, color='#111111')),
            xaxis_title="Time", yaxis_title="Quality (%)",
            hovermode='x unified', height=300,
            margin=dict(l=50,r=20,t=50,b=40),
            yaxis=dict(range=[0,102], gridcolor='#E5E7EB', tickfont=dict(color='#6B7280')),
            xaxis=dict(gridcolor='#E5E7EB', tickfont=dict(color='#6B7280')),
            plot_bgcolor='#F3F4F6', paper_bgcolor='#FFFFFF', font=dict(color='#111111'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                        font=dict(size=12, color='#374151'))
        )
        st.markdown('<div class="diagnostics-chart-wrapper">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    render_dashboard()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EVENT TIMELINE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Event Timeline":
    st.markdown('<style>.block-container{background:#FFFFFF!important;border-radius:12px;padding:28px 32px;}</style>', unsafe_allow_html=True)

    st.markdown(
        '<div style="margin-bottom:24px;">' +
        '<div style="font-size:11px;font-weight:600;color:#A1A1AA;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px;">AIRA · Logs</div>' +
        '<div style="font-size:22px;font-weight:700;color:#18181B;">Event Timeline</div>' +
        '</div>',
        unsafe_allow_html=True
    )

    log_df = load_log()

    if log_df.empty:
        st.markdown("""
            <div style="background:#FFFFFF;border-radius:12px;padding:48px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,0.08);">
                <div style="font-size:32px;margin-bottom:12px;">📭</div>
                <div style="font-size:16px;font-weight:600;color:#111111;margin-bottom:8px;">No events logged yet</div>
                <div style="font-size:14px;color:#6B7280;">Events are recorded when Simulate Live Data is turned on.</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        log_df = log_df.sort_values('timestamp', ascending=False)

        st.markdown('<div style="background:#FFFFFF;border-radius:12px;padding:20px 24px;border:1px solid #E4E4E7;box-shadow:0 1px 3px rgba(0,0,0,0.06);margin-bottom:20px;">', unsafe_allow_html=True)
        filter_col1, filter_col2, filter_col3 = st.columns([2,2,1])
        with filter_col1:
            conf_options = [
                "All",
                f"High (≥ {CONF_AMBER}%)",
                f"Medium ({CONF_RED}–{CONF_AMBER}%)",
                f"Low (< {CONF_RED}%)",
            ]
            conf_filter = st.selectbox("Confidence", conf_options)
        with filter_col2:
            all_emotions   = ["All"] + sorted(log_df['primary_emotion'].unique().tolist())
            emotion_filter = st.selectbox("Emotion", all_emotions)
        with filter_col3:
            st.markdown('<div style="padding-top:28px;">', unsafe_allow_html=True)
            if st.button("🗑 Clear Log", use_container_width=True):
                if os.path.exists(LOG_PATH):
                    os.remove(LOG_PATH)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        filtered = log_df.copy()
        if conf_filter == conf_options[1]:   filtered = filtered[filtered['confidence_band']=='high']
        elif conf_filter == conf_options[2]: filtered = filtered[filtered['confidence_band']=='medium']
        elif conf_filter == conf_options[3]: filtered = filtered[filtered['confidence_band']=='low']
        if emotion_filter != "All":            filtered = filtered[filtered['primary_emotion']==emotion_filter]

        total  = len(filtered)
        n_high = len(filtered[filtered['confidence_band']=='high'])
        n_med  = len(filtered[filtered['confidence_band']=='medium'])
        n_low  = len(filtered[filtered['confidence_band']=='low'])

        s1,s2,s3,s4 = st.columns(4)
        for col,label,val,color in [
            (s1,"Showing",total,"#111111"),
            (s2,"High Confidence",n_high,"#16A34A"),
            (s3,"Medium Confidence",n_med,"#D97706"),
            (s4,"Low Confidence",n_low,"#DC2626"),
        ]:
            with col:
                st.markdown(
                    f'<div style="background:#FFFFFF;border-radius:12px;padding:16px 20px;box-shadow:0 1px 4px rgba(0,0,0,0.08);margin-bottom:20px;">'
                    f'<div style="font-size:12px;color:#6B7280;margin-bottom:4px;">{label}</div>'
                    f'<div style="font-size:28px;font-weight:700;color:{color};">{val}</div>'
                    f'</div>', unsafe_allow_html=True
                )

        if filtered.empty:
            st.markdown('<div style="background:#FFFFFF;border-radius:12px;padding:32px;text-align:center;color:#6B7280;font-size:14px;">No events match the current filters.</div>', unsafe_allow_html=True)
        else:
            selected_row = None
            for _, ev in filtered.head(100).iterrows():
                ts   = pd.to_datetime(ev['timestamp']).strftime('%H:%M:%S')
                conf = float(ev['confidence'])
                c_col = conf_color(conf)

                ca,cb,cc,cd,ce = st.columns([1.5,2,2,1.5,0.8])
                with ca:
                    st.markdown(f'<div style="font-size:13px;font-weight:600;color:#111111;padding-top:10px;">{ts}</div>', unsafe_allow_html=True)
                with cb:
                    st.markdown(f'<div style="font-size:14px;font-weight:700;color:#111111;padding-top:10px;">{ev["primary_emotion"]}</div>', unsafe_allow_html=True)
                with cc:
                    st.markdown(
                        f'<div style="padding-top:6px;">'
                        f'<div style="font-size:12px;color:#6B7280;margin-bottom:4px;">Confidence</div>'
                        f'<div style="display:flex;align-items:center;gap:8px;">'
                        f'<div style="flex:1;background:#E5E7EB;border-radius:999px;height:6px;overflow:hidden;">'
                        f'<div style="background:{c_col};height:6px;border-radius:999px;width:{min(conf,100)}%;"></div>'
                        f'</div>'
                        f'<div style="font-size:13px;font-weight:600;color:{c_col};min-width:40px;">{conf:.1f}%</div>'
                        f'</div></div>', unsafe_allow_html=True
                    )
                with cd:
                    st.markdown(f'<div style="padding-top:10px;">{band_badge(ev["confidence_band"])}</div>', unsafe_allow_html=True)
                with ce:
                    if st.button("Inspect", key=f"ins_{ev['timestamp']}"):
                        selected_row = ev
                st.markdown('<hr style="border-color:#E5E7EB;margin:6px 0;">', unsafe_allow_html=True)

            if selected_row is not None:
                ev     = selected_row
                ts_fmt = pd.to_datetime(ev['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                st.markdown(f'<div style="margin-top:24px;font-size:15px;font-weight:700;color:#111111;margin-bottom:16px;">🔍 Diagnostic Detail — {ts_fmt}</div>', unsafe_allow_html=True)

                d1,d2 = st.columns(2)
                with d1:
                    st.markdown((
                        '<div style="background:#FFFFFF;border-radius:12px;padding:20px;box-shadow:0 1px 4px rgba(0,0,0,0.08);">'
                            '<div style="font-size:14px;font-weight:600;color:#374151;margin-bottom:16px;">Signal & Confidence</div>'
                            f'<div style="display:flex;justify-content:space-between;margin-bottom:10px;"><span style="font-size:13px;color:#6B7280;">Confidence</span><span style="font-size:13px;font-weight:600;color:{conf_color(float(ev["confidence"]))};">{float(ev["confidence"]):.1f}%</span></div>'
                            f'<div style="display:flex;justify-content:space-between;margin-bottom:10px;"><span style="font-size:13px;color:#6B7280;">Video Signal Quality</span><span style="font-size:13px;font-weight:600;color:#111111;">{ev["video_quality"]}%</span></div>'
                            f'<div style="display:flex;justify-content:space-between;margin-bottom:10px;"><span style="font-size:13px;color:#6B7280;">Audio Signal Quality</span><span style="font-size:13px;font-weight:600;color:#111111;">{ev["audio_quality"]}%</span></div>'
                            f'<div style="display:flex;justify-content:space-between;"><span style="font-size:13px;color:#6B7280;">Dominant Emotion</span><span style="font-size:13px;font-weight:600;color:#111111;">{ev["primary_emotion"]}</span></div>'
                        '</div>'
                    ), unsafe_allow_html=True)
                with d2:
                    el = ['Happy','Sad','Fear','Angry','Disgust','Neutral','Surprise']
                    ev_vals = [ev['happy'],ev['sad'],ev['fear'],ev['angry'],ev['disgust'],ev['neutral'],ev.get('surprise',0)]
                    fig_ev = go.Figure(go.Bar(
                        x=el, y=ev_vals,
                        marker_color=['#111111' if e==ev['primary_emotion'] else '#E5E7EB' for e in el],
                        text=[f'{v:.1f}%' for v in ev_vals], textposition='outside'
                    ))
                    fig_ev.update_layout(
                        title=dict(text="Emotion Scores at Event", font=dict(size=13,color='#111111')),
                        height=280, margin=dict(l=20,r=20,t=40,b=20),
                        plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF',
                        yaxis=dict(range=[0,110],gridcolor='#E5E7EB',tickfont=dict(color='#6B7280')),
                        xaxis=dict(tickfont=dict(color='#374151')),
                        showlegend=False, font=dict(color='#111111')
                    )
                    st.plotly_chart(fig_ev, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Model Performance":
    st.markdown('<style>.block-container{background:#FFFFFF!important;border-radius:12px;padding:28px 32px;}</style>', unsafe_allow_html=True)

    # Header + Refresh button
    header_col, btn_col = st.columns([8, 1])
    with header_col:
        st.markdown(
            '<div style="margin-bottom:24px;">' +
            '<div style="font-size:11px;font-weight:600;color:#A1A1AA;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px;">AIRA · ML Analytics</div>' +
            '<div style="font-size:22px;font-weight:700;color:#18181B;">Model Performance</div>' +
            '</div>',
            unsafe_allow_html=True
        )
    with btn_col:
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.conf_matrix = generate_confusion_matrix()
            st.session_state.latency = generate_latency()
            cpu_last = st.session_state.cpu_history[-1] if st.session_state.cpu_history else 50
            new_cpu = cpu_last + np.random.uniform(-6, 6)
            st.session_state.cpu_history = st.session_state.cpu_history + [max(5, min(90, round(new_cpu, 1)))]
            st.rerun()

    # Metrics from session state
    mat = st.session_state.conf_matrix
    acc = accuracy_from_matrix(mat) * 100
    f1 = f1_from_matrix(mat) * 100
    lat = st.session_state.latency
    cpu = st.session_state.cpu_history[-1] if st.session_state.cpu_history else 50
    cpu_h = st.session_state.cpu_history

    # Metric tiles
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f'<div style="background:#FFFFFF;border-radius:12px;padding:18px 20px;border:1px solid #E4E4E7;box-shadow:0 1px 3px rgba(0,0,0,0.06);">'
            f'<div style="font-size:11px;font-weight:600;color:#6B7280;margin-bottom:6px;">Accuracy</div>'
            f'<div style="font-size:24px;font-weight:700;color:#2563EB;">{acc:.1f}%</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with m2:
        st.markdown(
            f'<div style="background:#FFFFFF;border-radius:12px;padding:18px 20px;border:1px solid #E4E4E7;box-shadow:0 1px 3px rgba(0,0,0,0.06);">'
            f'<div style="font-size:11px;font-weight:600;color:#6B7280;margin-bottom:6px;">F1 Score</div>'
            f'<div style="font-size:24px;font-weight:700;color:#059669;">{f1:.1f}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with m3:
        st.markdown(
            f'<div style="background:#FFFFFF;border-radius:12px;padding:18px 20px;border:1px solid #E4E4E7;box-shadow:0 1px 3px rgba(0,0,0,0.06);">'
            f'<div style="font-size:11px;font-weight:600;color:#6B7280;margin-bottom:6px;">Latency</div>'
            f'<div style="font-size:24px;font-weight:700;color:#F59E0B;">{lat}ms</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with m4:
        cpu_color = '#DC2626' if cpu > 80 else '#F59E0B' if cpu > 60 else '#059669'
        st.markdown(
            f'<div style="background:#FFFFFF;border-radius:12px;padding:18px 20px;border:1px solid #E4E4E7;box-shadow:0 1px 3px rgba(0,0,0,0.06);">'
            f'<div style="font-size:11px;font-weight:600;color:#6B7280;margin-bottom:6px;">CPU Usage</div>'
            f'<div style="font-size:24px;font-weight:700;color:{cpu_color};">{cpu:.1f}%</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div style="margin:28px 0;"></div>', unsafe_allow_html=True)

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(mat, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - Emotion Classification', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    cm_b64 = fig_to_b64(fig)

    st.markdown(
        f'<div style="background:#FFFFFF;border-radius:12px;padding:24px;border:1px solid #E4E4E7;box-shadow:0 1px 3px rgba(0,0,0,0.06);">'
        f'<img src="data:image/png;base64,{cm_b64}" style="width:100%;height:auto;">'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div style="margin:20px 0;"></div>', unsafe_allow_html=True)

    # CPU History Chart
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(cpu_h, linewidth=2, color='#2563EB', alpha=0.8)
    ax.axhline(y=60, color='#F59E0B', linestyle='--', linewidth=1, alpha=0.5, label='Warning (60%)')
    ax.axhline(y=80, color='#DC2626', linestyle='--', linewidth=1, alpha=0.5, label='Critical (80%)')
    ax.fill_between(range(len(cpu_h)), 0, cpu_h, alpha=0.1, color='#2563EB')
    ax.set_ylim([0, 100])
    ax.set_xlabel('Time (seconds)', fontsize=10)
    ax.set_ylabel('CPU Usage (%)', fontsize=10)
    ax.set_title('CPU Usage - Last 60 Seconds', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    cpu_b64 = fig_to_b64(fig)

    # Latency scatter
    import random
    np.random.seed(42)
    latency_samples = [st.session_state.latency + np.random.normal(0, 5) for _ in range(30)]
    latency_samples = [max(25, min(95, l)) for l in latency_samples]
    
    fig, ax = plt.subplots(figsize=(10, 3))
    colors = ['#EF4444' if x > 80 else '#FBBF24' if x > 60 else '#34D399' for x in latency_samples]
    ax.scatter(range(len(latency_samples)), latency_samples, c=colors, alpha=0.6, s=30)
    ax.axhline(y=60, color='#F59E0B', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=80, color='#DC2626', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylim([20, 100])
    ax.set_xlabel('Sample #', fontsize=10)
    ax.set_ylabel('Latency (ms)', fontsize=10)
    ax.set_title('Inference Latency - Last 30 Samples', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    lat_b64 = fig_to_b64(fig)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f'<div style="background:#FFFFFF;border-radius:12px;padding:24px;border:1px solid #E4E4E7;box-shadow:0 1px 3px rgba(0,0,0,0.06);">'
            f'<img src="data:image/png;base64,{cpu_b64}" style="width:100%;height:auto;">'
            '</div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f'<div style="background:#FFFFFF;border-radius:12px;padding:24px;border:1px solid #E4E4E7;box-shadow:0 1px 3px rgba(0,0,0,0.06);">'
            f'<img src="data:image/png;base64,{lat_b64}" style="width:100%;height:auto;">'
            '</div>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Settings":
    st.markdown('<style>.block-container{background:#FFFFFF!important;border-radius:12px;padding:28px 32px;}</style>', unsafe_allow_html=True)

    st.markdown(
        '<div style="margin-bottom:24px;">' +
        '<div style="font-size:11px;font-weight:600;color:#A1A1AA;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px;">AIRA · Configuration</div>' +
        '<div style="font-size:22px;font-weight:700;color:#18181B;">Settings</div>' +
        '</div>',
        unsafe_allow_html=True
    )

    current = load_config()

    # Style st.container() blocks on this page as white cards
    st.markdown("""
        <style>
        /* Target all direct stVerticalBlock children on settings page as white cards */
        section.main .block-container > div > div > div > div > [data-testid="stVerticalBlock"] > div > [data-testid="stVerticalBlock"] {
            background: #FFFFFF !important;
            border-radius: 12px !important;
            border: 1px solid #E4E4E7 !important;
            padding: 24px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
            margin-bottom: 12px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # ── PERFORMANCE CARD ──────────────────────────────────────────────────────
    with st.container(border=False):
        st.markdown(
            '<div style="font-size:16px;font-weight:700;color:#111111;margin-bottom:2px;">Performance</div>'
            '<div style="font-size:13px;color:#6B7280;margin-bottom:12px;">Adjust based on the processing capacity of the robot\'s chip.</div>',
            unsafe_allow_html=True
        )
        new_refresh  = st.slider("Dashboard refresh interval (seconds)", 1, 30, current["refresh_interval"],
                                  help="Lower = more responsive but more CPU.")
        new_log_freq = st.slider("Log every N frames", 1, 20, current["log_frequency"],
                                  help="1 = every frame. Higher = less storage.")

    # ── CONFIDENCE THRESHOLDS CARD ────────────────────────────────────────────
    with st.container(border=False):
        st.markdown(
            '<div style="font-size:16px;font-weight:700;color:#111111;margin-bottom:2px;">Confidence Thresholds</div>'
            '<div style="font-size:13px;color:#6B7280;margin-bottom:12px;">Define what counts as low, medium, and high confidence for your model.</div>',
            unsafe_allow_html=True
        )
        new_conf_red   = st.slider("Red threshold — low confidence (%)", 10, 60, current["conf_red"])
        new_conf_amber = st.slider("Amber threshold — degraded warning (%)", new_conf_red+5, 90,
                                    max(current["conf_amber"], new_conf_red+5))
        new_recovery   = st.slider("Recovery frames to close a low-confidence event", 1, 10, current["recovery_frames"])

    # ── SAVE ──────────────────────────────────────────────────────────────────
    col_save, col_status = st.columns([1, 3])
    with col_save:
        if st.button("💾 Save Settings", type="primary", use_container_width=True):
            save_config({
                "refresh_interval": new_refresh,
                "log_frequency":    new_log_freq,
                "conf_red":         new_conf_red,
                "conf_amber":       new_conf_amber,
                "recovery_frames":  new_recovery,
            })
            st.success("Settings saved. Restart the app for changes to take full effect.")

    st.markdown(
        f'<div style="background:#FFFFFF;border-radius:8px;padding:14px 18px;font-size:12px;'
        f'color:#6B7280;margin-top:8px;border:1px solid #E5E7EB;">'
        f'<strong style="color:#374151;">Currently active:</strong> '
        f'Refresh {REFRESH_INTERVAL}s · Log every {LOG_FREQUENCY} frame(s) · '
        f'Red &lt;{CONF_RED}% · Amber &lt;{CONF_AMBER}% · Recovery {RECOVERY_FRAMES} frames'
        f'</div>',
        unsafe_allow_html=True
    )