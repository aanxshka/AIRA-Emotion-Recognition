import streamlit as st
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import deque

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Dashboard",
    page_icon="🧠",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

  html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

  .stApp { background: #0f1117; color: #e8eaf0; }

  .card {
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 0;
  }

  .card-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.3px;
    margin-bottom: 18px;
  }

  .metric-row { display: flex; gap: 16px; margin-bottom: 24px; }
  .metric-chip {
    background: #23263a;
    border: 1px solid #2e3148;
    border-radius: 10px;
    padding: 12px 20px;
    flex: 1;
  }
  .metric-label {
    font-size: 0.72rem;
    color: #7b80a0;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 4px;
  }
  .metric-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #e8eaf0;
    font-family: 'JetBrains Mono', monospace;
  }
  .metric-unit { font-size: 0.85rem; color: #7b80a0; margin-left: 4px; }

  .accent-green { color: #34d399; }
  .accent-blue  { color: #60a5fa; }

  .section-label {
    font-size: 0.7rem;
    color: #5a5f7a;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
  }

  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 2rem; padding-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
N = len(EMOTIONS)

# ── Data generators ────────────────────────────────────────────────────────────

def generate_confusion_matrix():
    mat = np.zeros((N, N), dtype=int)
    for i in range(N):
        total = random.randint(80, 120)
        correct = int(total * random.uniform(0.55, 0.80))
        mat[i, i] = correct
        remaining = total - correct
        others = [j for j in range(N) if j != i]
        for j in others[:-1]:
            v = random.randint(0, remaining)
            mat[i, j] = v
            remaining -= v
            if remaining <= 0:
                break
        mat[i, others[-1]] = max(0, remaining)
    return mat


def accuracy_from_matrix(mat):
    return mat.diagonal().sum() / mat.sum()


def f1_from_matrix(mat):
    scores = []
    for i in range(N):
        tp = mat[i, i]
        fp = mat[:, i].sum() - tp
        fn = mat[i, :].sum() - tp
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom else 0.0)
    return float(np.mean(scores))


def generate_cpu_history(n=60):
    base = random.uniform(40, 60)
    hist = []
    for _ in range(n):
        base += random.uniform(-3, 3)
        base = max(10, min(95, base))
        hist.append(round(base, 1))
    return hist


def generate_latency():
    return random.randint(30, 80)


# ── Session state ──────────────────────────────────────────────────────────────
if "cpu_history" not in st.session_state:
    st.session_state.cpu_history = deque(generate_cpu_history(), maxlen=60)
if "conf_matrix" not in st.session_state:
    st.session_state.conf_matrix = generate_confusion_matrix()
if "latency" not in st.session_state:
    st.session_state.latency = generate_latency()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    auto_refresh = st.toggle("Auto-refresh (2 s)", value=False)
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.session_state.conf_matrix = generate_confusion_matrix()
        st.session_state.latency = generate_latency()
        new_cpu = st.session_state.cpu_history[-1] + random.uniform(-5, 5)
        st.session_state.cpu_history.append(max(10, min(95, round(new_cpu, 1))))
        st.rerun()
    st.markdown("---")
    st.caption("Data is randomly generated for prototyping.")

if auto_refresh:
    time.sleep(2)
    new_cpu = st.session_state.cpu_history[-1] + random.uniform(-5, 5)
    st.session_state.cpu_history.append(max(10, min(95, round(new_cpu, 1))))
    st.session_state.latency = generate_latency()
    st.rerun()

# ── Derived metrics ────────────────────────────────────────────────────────────
mat   = st.session_state.conf_matrix
acc   = accuracy_from_matrix(mat) * 100
f1    = f1_from_matrix(mat) * 100
lat   = st.session_state.latency
cpu   = st.session_state.cpu_history[-1]
cpu_h = list(st.session_state.cpu_history)

# ── Two-column layout ──────────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

# ═══════════════════════════════════════════════
# LEFT: Model Performance
# ═══════════════════════════════════════════════
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Model Performance</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-chip">
        <div class="metric-label">Overall Accuracy</div>
        <div class="metric-value accent-green">{acc:.1f}<span class="metric-unit">%</span></div>
      </div>
      <div class="metric-chip">
        <div class="metric-label">F1 Score</div>
        <div class="metric-value accent-blue">{f1:.1f}<span class="metric-unit">%</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Confusion Matrix</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#1a1d27")
    ax.set_facecolor("#1a1d27")

    sns.heatmap(
        mat,
        annot=True,
        fmt="d",
        cmap="viridis",
        linewidths=0.4,
        linecolor="#0f1117",
        xticklabels=EMOTIONS,
        yticklabels=EMOTIONS,
        ax=ax,
        annot_kws={"size": 8, "color": "white"},
        cbar=False,
    )
    ax.set_xlabel("Predicted label", color="#7b80a0", fontsize=9, labelpad=8)
    ax.set_ylabel("True label",      color="#7b80a0", fontsize=9, labelpad=8)
    ax.tick_params(colors="#7b80a0", labelsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# RIGHT: System Performance
# ═══════════════════════════════════════════════
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">System Performance</div>', unsafe_allow_html=True)

    lat_color = "#34d399" if lat < 50 else "#fbbf24" if lat < 70 else "#f87171"
    cpu_color = "#34d399" if cpu < 60 else "#fbbf24" if cpu < 80 else "#f87171"

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-chip">
        <div class="metric-label">Latency</div>
        <div class="metric-value" style="color:{lat_color}">{lat}<span class="metric-unit">ms</span></div>
        <div class="metric-label" style="margin-top:2px;">Processing Time</div>
      </div>
      <div class="metric-chip">
        <div class="metric-label">CPU Usage</div>
        <div class="metric-value" style="color:{cpu_color}">{cpu:.0f}<span class="metric-unit">%</span></div>
        <div class="metric-label" style="margin-top:2px;">Current Load</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">CPU Usage Over Time</div>', unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(6, 2.8))
    fig2.patch.set_facecolor("#1a1d27")
    ax2.set_facecolor("#1a1d27")

    xs = list(range(len(cpu_h)))
    ax2.plot(xs, cpu_h, color="#60a5fa", linewidth=2, solid_capstyle="round")
    ax2.fill_between(xs, cpu_h, alpha=0.12, color="#60a5fa")

    ax2.axhline(80, color="#f87171", linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.axhline(60, color="#fbbf24", linewidth=0.8, linestyle="--", alpha=0.6)

    ax2.set_xlim(0, max(len(cpu_h) - 1, 1))
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("%", color="#7b80a0", fontsize=9)
    ax2.tick_params(colors="#7b80a0", labelsize=8)
    ax2.set_xticks([0, len(cpu_h) - 1])
    ax2.set_xticklabels(["60 s ago", "now"], color="#7b80a0", fontsize=8)
    for spine in ax2.spines.values():
        spine.set_color("#2a2d3a")

    handles = [
        mpatches.Patch(color="#fbbf24", alpha=0.8, label="60% warn"),
        mpatches.Patch(color="#f87171", alpha=0.8, label="80% crit"),
    ]
    ax2.legend(handles=handles, fontsize=7, facecolor="#1a1d27",
               edgecolor="#2a2d3a", labelcolor="#7b80a0", loc="upper left")

    plt.tight_layout(pad=0.5)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    st.markdown('</div>', unsafe_allow_html=True)