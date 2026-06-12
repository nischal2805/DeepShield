"""
DeepShield Flask Web App — Face-Swap Deepfake Detector + AI-Gen Detector
Single-file, no separate static/ needed.

Run:
    E:\\wic\\deepshield\\.venv\\Scripts\\python.exe flask_app.py
Then open http://127.0.0.1:5000
"""

# ── CRITICAL: set env vars and import native libs BEFORE torch/models ──────────
import os
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import cv2  # noqa: E402 — must come before torch on Windows
import torch  # noqa: E402
from inference.pipeline import DeepShieldPipeline, _TRANSFORM  # noqa: E402
# models.* imports MUST come after inference.pipeline (Windows 0xC0000005 rule)
from models.aigen_branch import AIGenDetector  # noqa: E402
# ────────────────────────────────────────────────────────────────────────────────

import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ── Checkpoint paths ────────────────────────────────────────────────────────────
CKPT_NEW    = 'D:/deepshield_data/checkpoints/wavelet_clip/wavelet_clip_best_auc.pth'
CKPT_OLD    = 'D:/deepshield_data/checkpoints/wavelet_clip/wavelet_clip_OLD_leaky.pth'
FC_PATH     = 'D:/deepshield_data/checkpoints/aigen/fc_weights.pth'
THRESHOLD   = 0.15          # face-swap threshold (Models 1 & 2)
AIGEN_THRESHOLD = 0.10      # AI-gen probe threshold (Model 3)

# ── Load both pipelines + AI-gen detector once at startup ───────────────────────
print("=" * 60)
print("DeepShield — loading pipelines at startup …")
print("=" * 60)

PIPELINES = {}

print("\n[1/3] Loading Model 1 (new, recommended) …")
PIPELINES['1'] = DeepShieldPipeline(wclip_ckpt=CKPT_NEW)

print("\n[2/3] Loading Model 2 (old) …")
PIPELINES['2'] = DeepShieldPipeline(wclip_ckpt=CKPT_OLD)

# Reuse CLIP already loaded in Model 1 — never call from_pretrained again
print("\n[3/3] Loading Model 3 — AI-Gen detector (UniversalFakeDetect probe) …")
_wclip_m1    = PIPELINES['1'].ensemble.wclip          # WaveletCLIP instance
_clip_handle = _wclip_m1.clip_projection.clip          # transformers CLIPModel (real weights)
AIGEN_DETECTOR = AIGenDetector(
    clip_model=_clip_handle,
    to_clip_norm_fn=_wclip_m1._to_clip_norm,
    device=PIPELINES['1'].device,
    fc_path=FC_PATH,
    threshold=AIGEN_THRESHOLD,
    use_l2_norm=False,          # raw feats: AUC=0.8956, F1=0.7826 on validation set
)

print("\n[OK] All three models ready.\n")

# ── Flask app ────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins=['http://localhost:5173', 'http://127.0.0.1:5173', 'http://localhost:3001'])
# ── Routes ───────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    import time

    # ── Validate inputs ──────────────────────────────────────────────────────────
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    model_id = request.form.get('model_id', '1')
    if model_id not in PIPELINES and model_id != '3':
        return jsonify({'error': f'Unknown model_id: {model_id}'}), 400

    # ── Decode image directly from bytes (no disk write) ────────────────────────
    file_bytes = file.read()
    arr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({'error': 'Cannot decode image. Please upload a valid JPEG/PNG.'}), 400

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── Run inference ────────────────────────────────────────────────────────────
    t0 = time.perf_counter()

    if model_id == '3':
        # ── Model 3: AI-Gen detector (UniversalFakeDetect CLIP linear probe) ──
        # Reuse the face detector from pipeline 1 (detector is stateless)
        p_ref = PIPELINES['1']
        faces = p_ref.detector.detect(img_rgb)
        if not faces:
            return jsonify({
                'verdict':     'NO FACE',
                'aigen_score': None,
                'model_id':    model_id,
                'threshold':   AIGEN_THRESHOLD,
                'latency_ms':  round((time.perf_counter() - t0) * 1000, 1),
            })

        crop = p_ref.detector.align_and_crop(img_rgb, faces[0], size=224)
        t    = _TRANSFORM(crop).unsqueeze(0).to(p_ref.device)

        with torch.no_grad():
            aigen_score = AIGEN_DETECTOR.predict(t)

        verdict    = 'AI-GENERATED' if aigen_score > AIGEN_THRESHOLD else 'REAL'
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        return jsonify({
            'verdict':     verdict,
            'aigen_score': round(aigen_score, 6),
            'model_id':    model_id,
            'threshold':   AIGEN_THRESHOLD,
            'latency_ms':  latency_ms,
        })

    else:
        # ── Models 1 & 2: face-swap detector (Wavelet-CLIP ensemble) ──────────
        p = PIPELINES[model_id]

        faces = p.detector.detect(img_rgb)
        if not faces:
            return jsonify({
                'verdict':     'NO FACE DETECTED',
                'wclip_score': None,
                'model_id':    model_id,
                'threshold':   THRESHOLD,
                'latency_ms':  round((time.perf_counter() - t0) * 1000, 1),
            })

        crop = p.detector.align_and_crop(img_rgb, faces[0], size=224)
        t    = _TRANSFORM(crop).unsqueeze(0).to(p.device)

        with torch.no_grad():
            _f, _sb, wc = p.ensemble(t)

        wclip_score = float(wc[0])
        verdict     = 'FACE-SWAP DEEPFAKE' if wclip_score > THRESHOLD else 'REAL'
        latency_ms  = round((time.perf_counter() - t0) * 1000, 1)

        return jsonify({
            'verdict':     verdict,
            'wclip_score': round(wclip_score, 6),
            'model_id':    model_id,
            'threshold':   THRESHOLD,
            'latency_ms':  latency_ms,
        })


# ── Entry point ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
