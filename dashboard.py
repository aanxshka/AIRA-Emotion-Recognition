import streamlit as st
import numpy as np
import random
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import deque
import io, base64

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ML Dashboard", page_icon="🧠", layout="wide")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #e8ecf3; }
.block-container { padding: 2rem 2rem !important; max-width: 100% !important; }
#MainMenu, footer, header { visibility: hidden; }

/* Hide sidebar toggle button entirely */
[data-testid="collapsedControl"] { display: none; }
section[data-testid="stSidebar"] { display: none; }

.card {
    background: #ffffff;
    border-radius: 18px;
    padding: 28px 28px 24px 28px;
    box-shadow: 0 1px 8px rgba(0,0,0,0.08);
    width: 100%;
    box-sizing: border-box;
}
.card-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #111827;
    margin: 0 0 20px 0;
}
.metric-row { display: flex; gap: 14px; margin-bottom: 20px; }
.metric-chip {
    flex: 1;
    background: #f3f4f6;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 14px 18px 12px 18px;
}
.chip-label {
    font-size: 0.67rem;
    font-weight: 600;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.9px;
    margin-bottom: 4px;
}
.chip-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.1;
}
.chip-unit { font-size: 0.95rem; font-weight: 600; margin-left: 1px; }
.chip-sub {
    font-size: 0.67rem;
    font-weight: 600;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.9px;
    margin-top: 5px;
}
.section-label {
    font-size: 0.67rem;
    font-weight: 600;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.9px;
    margin-bottom: 12px;
}
.chart-img {
    width: 100%;
    border-radius: 10px;
    display: block;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
N = len(EMOTIONS)

# ── Helpers ────────────────────────────────────────────────────────────────────
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ── Generators ─────────────────────────────────────────────────────────────────
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
    base = random.uniform(30, 50)
    hist = []
    for _ in range(n):
        base += random.uniform(-4, 4)
        base = max(5, min(90, base))
        hist.append(round(base, 1))
    return hist

def generate_latency():
    return random.randint(30, 90)

# ── Session state ──────────────────────────────────────────────────────────────
if "cpu_history" not in st.session_state:
    st.session_state.cpu_history = deque(generate_cpu_history(), maxlen=60)
if "conf_matrix" not in st.session_state:
    st.session_state.conf_matrix = generate_confusion_matrix()
if "latency" not in st.session_state:
    st.session_state.latency = generate_latency()

# ── Refresh button (top-right, no sidebar) ─────────────────────────────────────
_, btn_col = st.columns([8, 1])
with btn_col:
    if st.button("🔄 Refresh", use_container_width=True):
        st.session_state.conf_matrix = generate_confusion_matrix()
        st.session_state.latency = generate_latency()
        new_cpu = st.session_state.cpu_history[-1] + random.uniform(-6, 6)
        st.session_state.cpu_history.append(max(5, min(90, round(new_cpu, 1))))
        st.rerun()

# ── Derived metrics ────────────────────────────────────────────────────────────
mat   = st.session_state.conf_matrix
acc   = accuracy_from_matrix(mat) * 100
f1    = f1_from_matrix(mat) * 100
lat   = st.session_state.latency
cpu   = st.session_state.cpu_history[-1]
cpu_h = list(st.session_state.cpu_history)

lat_color = "#16a34a" if lat < 50 else "#d97706" if lat < 70 else "#dc2626"
cpu_color = "#16a34a" if cpu < 60 else "#d97706" if cpu < 80 else "#dc2626"

# ── Confusion matrix: smart white/dark text per cell ──────────────────────────
fig1, ax1 = plt.subplots(figsize=(7, 6))
fig1.patch.set_facecolor("#ffffff")
ax1.set_facecolor("#ffffff")

# Normalise to [0,1] to decide text colour threshold
mat_norm = mat / (mat.max() + 1e-9)

# Draw heatmap without annotations first
sns.heatmap(
    mat,
    annot=False,
    cmap="Blues",
    linewidths=0.5,
    linecolor="#f0f2f5",
    xticklabels=EMOTIONS,
    yticklabels=EMOTIONS,
    ax=ax1,
    cbar=False,
)

# Add annotations manually with adaptive colour
for i in range(N):
    for j in range(N):
        val = mat[i, j]
        brightness = mat_norm[i, j]
        # Use white text on dark cells (brightness > 0.45), dark text on light cells
        text_color = "#ffffff" if brightness > 0.45 else "#374151"
        ax1.text(
            j + 0.5, i + 0.5, str(val),
            ha="center", va="center",
            fontsize=9, color=text_color, fontweight="600"
        )

ax1.set_xlabel("Predicted label", color="#6b7280", fontsize=9, labelpad=10)
ax1.set_ylabel("True label",      color="#6b7280", fontsize=9, labelpad=10)
ax1.tick_params(colors="#6b7280", labelsize=8.5, length=0)
plt.xticks(rotation=45, ha="right", fontsize=8.5, color="#6b7280")
plt.yticks(rotation=0, fontsize=8.5, color="#6b7280")
for spine in ax1.spines.values():
    spine.set_visible(False)
fig1.tight_layout(pad=0.8)
cm_b64 = fig_to_b64(fig1)
plt.close(fig1)

# ── CPU sparkline ──────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(7, 3.2))
fig2.patch.set_facecolor("#ffffff")
ax2.set_facecolor("#ffffff")

xs = list(range(len(cpu_h)))
ax2.plot(xs, cpu_h, color="#2563eb", linewidth=2, solid_capstyle="round", zorder=3)
ax2.fill_between(xs, cpu_h, alpha=0.10, color="#2563eb", zorder=2)
ax2.axhline(80, color="#ef4444", linewidth=1.0, linestyle="--", alpha=0.8, zorder=1)
ax2.axhline(60, color="#f59e0b", linewidth=1.0, linestyle="--", alpha=0.8, zorder=1)

ax2.set_xlim(0, max(len(cpu_h) - 1, 1))
ax2.set_ylim(0, 105)
ax2.set_ylabel("%", color="#9ca3af", fontsize=9)
ax2.tick_params(colors="#9ca3af", labelsize=8, length=0)
ax2.yaxis.set_ticks([0, 20, 40, 60, 80, 100])
ax2.set_xticks([0, len(cpu_h) - 1])
ax2.set_xticklabels(["60 s ago", "now"], color="#9ca3af", fontsize=8.5)

for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax2.spines[spine].set_color("#e5e7eb")
ax2.grid(axis="y", color="#f3f4f6", linewidth=0.8, zorder=0)

handles = [
    mpatches.Patch(color="#f59e0b", alpha=0.9, label="60% warn"),
    mpatches.Patch(color="#ef4444", alpha=0.9, label="80% crit"),
]
ax2.legend(handles=handles, fontsize=8, facecolor="#ffffff",
           edgecolor="#e5e7eb", labelcolor="#6b7280",
           loc="upper right", framealpha=1)

fig2.tight_layout(pad=0.8)
cpu_b64 = fig_to_b64(fig2)
plt.close(fig2)

# ── Render ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(f"""
    <div class="card">
      <div class="card-title">Model Performance on RAVDEES dataset</div>
      <div class="metric-row">
        <div class="metric-chip">
          <div class="chip-label">Overall Accuracy</div>
          <div class="chip-value" style="color:#16a34a">{acc:.1f}<span class="chip-unit">%</span></div>
        </div>
        <div class="metric-chip">
          <div class="chip-label">F1 Score</div>
          <div class="chip-value" style="color:#2563eb">{f1:.1f}<span class="chip-unit">%</span></div>
        </div>
      </div>
      <div class="section-label">Confusion Matrix</div>
      <img class="chart-img" src="data:image/png;base64,{cm_b64}" />
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
      <div class="card-title">System Performance</div>
      <div class="metric-row">
        <div class="metric-chip">
          <div class="chip-label">Latency</div>
          <div class="chip-value" style="color:{lat_color}">{lat}<span class="chip-unit" style="color:{lat_color}">ms</span></div>
          <div class="chip-sub">Processing Time</div>
        </div>
        <div class="metric-chip">
          <div class="chip-label">CPU Usage</div>
          <div class="chip-value" style="color:{cpu_color}">{cpu:.0f}<span class="chip-unit" style="color:{cpu_color}">%</span></div>
          <div class="chip-sub">Current Load</div>
        </div>
      </div>
      <div class="section-label">CPU Usage Over Time</div>
      <img class="chart-img" src="data:image/png;base64,{cpu_b64}" />
    </div>
    """, unsafe_allow_html=True)
