"""
DeepShield — Streamlit inference app.
Upload an image → get deepfake/AI-generated verdict + Grad-CAM heatmap.

Run:
    streamlit run app.py

Checkpoints must exist (train first — see HOW_TO_RUN.md).
Set SBI_CKPT / WCLIP_CKPT / WEIGHTS env vars to override default paths.
"""
import base64
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepShield",
    page_icon="🛡️",
    layout="centered",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.verdict-fake  { background:#ff4b4b; color:white; padding:8px 18px; border-radius:8px; font-size:1.4rem; font-weight:bold; display:inline-block; }
.verdict-real  { background:#21c45e; color:white; padding:8px 18px; border-radius:8px; font-size:1.4rem; font-weight:bold; display:inline-block; }
.verdict-ai    { background:#f59e0b; color:white; padding:8px 18px; border-radius:8px; font-size:1.4rem; font-weight:bold; display:inline-block; }
.verdict-unk   { background:#6b7280; color:white; padding:8px 18px; border-radius:8px; font-size:1.4rem; font-weight:bold; display:inline-block; }
.score-box     { border:1px solid #ddd; border-radius:6px; padding:12px 16px; margin:4px 0; }
</style>
""", unsafe_allow_html=True)


# ── Cached pipeline loader ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_pipeline():
    from inference.pipeline import DeepShieldPipeline
    sbi_ckpt   = os.environ.get('SBI_CKPT')
    wclip_ckpt = os.environ.get('WCLIP_CKPT')
    weights    = os.environ.get('WEIGHTS')
    return DeepShieldPipeline(
        sbi_ckpt=sbi_ckpt,
        wclip_ckpt=wclip_ckpt,
        weights=weights,
    )


def verdict_html(verdict: str) -> str:
    css = {
        'FAKE':         'verdict-fake',
        'REAL':         'verdict-real',
        'AI-GENERATED': 'verdict-ai',
        'UNKNOWN':      'verdict-unk',
    }.get(verdict, 'verdict-unk')
    label = {'FAKE': '⚠ FAKE', 'REAL': '✓ REAL', 'AI-GENERATED': '🤖 AI-GENERATED', 'UNKNOWN': '? UNKNOWN'}.get(verdict, verdict)
    return f'<span class="{css}">{label}</span>'


def confidence_bar(score: float, label: str) -> None:
    pct = int(score * 100)
    color = '#ff4b4b' if score > 0.5 else '#21c45e'
    st.markdown(
        f'<div class="score-box">'
        f'<b>{label}</b><br>'
        f'<div style="background:#eee;border-radius:4px;height:12px;margin-top:6px;">'
        f'<div style="background:{color};width:{pct}%;height:12px;border-radius:4px;"></div>'
        f'</div>'
        f'<small>{pct}%</small>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🛡️ DeepShield")
st.caption("Deepfake & AI-generated image detection")

# Sidebar: model status
with st.sidebar:
    st.header("Model Status")
    pipeline = None
    error_msg = None
    try:
        pipeline = load_pipeline()
        st.success("Models loaded")
        if pipeline:
            w_sbi   = float(pipeline.ensemble.w_sbi)
            w_wclip = float(pipeline.ensemble.w_wclip)
            st.caption(f"Ensemble: SBI={w_sbi:.2f}  WCLIP={w_wclip:.2f}")
    except FileNotFoundError as e:
        error_msg = str(e)
        st.error("Checkpoints not found")
        with st.expander("Details"):
            st.code(error_msg)
    except Exception as e:
        error_msg = str(e)
        st.error("Load error")
        with st.expander("Details"):
            st.code(error_msg)

    st.divider()
    st.caption("**Env vars** (optional overrides):")
    st.code("SBI_CKPT=path/to/sbi_best_auc.pth\nWCLIP_CKPT=path/to/wavelet_clip_best_auc.pth\nWEIGHTS=checkpoints/ensemble_weights.json")
    threshold = st.slider("Detection threshold", 0.1, 0.9, 0.5, 0.05)
    if pipeline:
        pipeline.threshold = threshold

# Main content
if pipeline is None:
    st.warning("⚠️ Models not loaded. Train first or set correct checkpoint paths. See `HOW_TO_RUN.md`.")
    if error_msg:
        st.error(error_msg)
    st.stop()

uploaded = st.file_uploader(
    "Upload image (JPG / PNG / WebP, max 10 MB)",
    type=['jpg', 'jpeg', 'png', 'webp'],
    accept_multiple_files=False,
)

if uploaded is None:
    st.info("Upload an image to begin detection.")
    st.stop()

file_size_mb = len(uploaded.getvalue()) / 1e6
if file_size_mb > 10:
    st.error(f"File too large ({file_size_mb:.1f} MB). Max 10 MB.")
    st.stop()

# Display uploaded image
pil_img = Image.open(uploaded).convert('RGB')
img_rgb = np.array(pil_img)

col_img, col_info = st.columns([1, 1])
with col_img:
    st.image(pil_img, caption="Uploaded image", use_container_width=True)
with col_info:
    st.caption(f"Size: {pil_img.width}×{pil_img.height}  |  {file_size_mb:.2f} MB")

st.divider()

# ── Run detection ─────────────────────────────────────────────────────────────
with st.spinner("Analyzing…"):
    try:
        result = pipeline.detect_array(img_rgb)
    except Exception as e:
        st.error(f"Detection error: {e}")
        st.exception(e)
        st.stop()

# ── Results ───────────────────────────────────────────────────────────────────
st.subheader("Result")

col_v, col_lat = st.columns([3, 1])
with col_v:
    st.markdown(verdict_html(result['verdict']), unsafe_allow_html=True)
with col_lat:
    st.metric("Latency", f"{result.get('latency_ms', 0):.0f} ms")

st.markdown("<br>", unsafe_allow_html=True)

track = result.get('track', 1)

if track == 1:
    col_a, col_b = st.columns(2)
    with col_a:
        confidence_bar(result['confidence'], "Overall confidence (fake)")
    with col_b:
        pass  # spacer

    with st.expander("Score breakdown"):
        confidence_bar(result['score_sbi'],   "SBI branch (EfficientNet-B4)")
        confidence_bar(result['score_wclip'], "Wavelet-CLIP branch")
        st.caption(f"Ensemble weights: SBI={float(pipeline.ensemble.w_sbi):.2f}  WCLIP={float(pipeline.ensemble.w_wclip):.2f}")

    # Grad-CAM heatmap
    heatmap_b64 = result.get('heatmap_b64')
    if heatmap_b64:
        st.subheader("Grad-CAM Heatmap")
        st.caption("Red regions indicate forgery artifacts detected by the SBI branch.")
        heatmap_bytes = base64.b64decode(heatmap_b64)
        heatmap_img   = Image.open(__import__('io').BytesIO(heatmap_bytes))
        st.image(heatmap_img, caption="Grad-CAM overlay on face crop", use_container_width=True)
    else:
        st.info("Grad-CAM unavailable (pytorch-grad-cam not installed).")

    face_bbox = result.get('face_bbox')
    if face_bbox:
        st.caption(f"Face detected at bbox: {face_bbox}")

else:
    # Track 2
    st.info("No face detected → AI-generation check (Track 2)")
    confidence_bar(result.get('confidence', 0.5), "AI-generated confidence")

    with st.expander("Track 2 details"):
        c2pa    = result.get('c2pa', {})
        synthid = result.get('synthid', {})

        st.markdown("**C2PA metadata:**")
        if c2pa.get('error'):
            st.warning(f"C2PA: {c2pa['error']}")
        else:
            st.json(c2pa)

        st.markdown("**SynthID watermark:**")
        if synthid.get('error'):
            st.warning(f"SynthID: {synthid['error']}")
            st.caption("Set `SYNTHID_API_KEY` and `SYNTHID_PROJECT` env vars to enable.")
        else:
            st.json(synthid)

# ── Raw JSON ──────────────────────────────────────────────────────────────────
with st.expander("Raw detection output"):
    display_result = {k: v for k, v in result.items() if k != 'heatmap_b64'}
    st.json(display_result)
