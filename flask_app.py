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
from flask import Flask, request, jsonify, render_template_string

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

# ── HTML template (inline CSS + JS) ─────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>DeepShield — Face-Swap Detector</title>
  <style>
    /* ── Reset & base ── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:          #18191c;
      --surface:     #23252b;
      --surface2:    #2c2f38;
      --border:      #35383f;
      --text:        #e4e6eb;
      --text-muted:  #8a8d96;
      --accent:      #c97f2a;        /* amber — NOT blue */
      --accent-h:    #e0952e;
      --red:         #e05252;
      --green:       #3ec97a;
      --grey:        #8a8d96;
      --radius:      12px;
      --radius-sm:   7px;
      --transition:  0.2s ease;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 2rem 1rem 4rem;
    }

    /* ── Header ── */
    header {
      text-align: center;
      margin-bottom: 2.5rem;
    }
    header h1 {
      font-size: clamp(1.6rem, 4vw, 2.4rem);
      font-weight: 700;
      letter-spacing: -0.5px;
      color: var(--text);
    }
    header h1 span { color: var(--accent); }
    header p {
      margin-top: 0.4rem;
      color: var(--text-muted);
      font-size: 0.95rem;
    }

    /* ── Card ── */
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.8rem;
      width: 100%;
      max-width: 560px;
      margin-bottom: 1.4rem;
    }

    /* ── Section label ── */
    .label {
      font-size: 0.78rem;
      font-weight: 600;
      letter-spacing: 0.8px;
      text-transform: uppercase;
      color: var(--text-muted);
      margin-bottom: 0.7rem;
    }

    /* ── Model selector ── */
    .model-selector {
      display: flex;
      gap: 0.6rem;
    }
    .model-btn {
      flex: 1;
      padding: 0.7rem 0.5rem;
      border-radius: var(--radius-sm);
      border: 1.5px solid var(--border);
      background: var(--surface2);
      color: var(--text-muted);
      font-size: 0.85rem;
      font-weight: 500;
      cursor: pointer;
      transition: all var(--transition);
      text-align: center;
      user-select: none;
    }
    .model-btn:hover { border-color: var(--accent); color: var(--text); }
    .model-btn.active {
      border-color: var(--accent);
      background: rgba(201,127,42,0.12);
      color: var(--accent);
      font-weight: 600;
    }
    .model-btn .tag {
      display: block;
      font-size: 0.7rem;
      color: var(--text-muted);
      margin-top: 2px;
      font-weight: 400;
    }
    .model-btn.active .tag { color: var(--accent); opacity: 0.75; }

    /* ── Drop zone ── */
    #drop-zone {
      border: 2px dashed var(--border);
      border-radius: var(--radius-sm);
      padding: 2.8rem 1.5rem;
      text-align: center;
      cursor: pointer;
      transition: all var(--transition);
      background: var(--surface2);
      position: relative;
    }
    #drop-zone:hover, #drop-zone.drag-over {
      border-color: var(--accent);
      background: rgba(201,127,42,0.06);
    }
    #drop-zone .icon { font-size: 2.4rem; margin-bottom: 0.6rem; }
    #drop-zone p { color: var(--text-muted); font-size: 0.9rem; }
    #drop-zone p strong { color: var(--text); }
    #file-input { display: none; }

    /* ── Preview ── */
    #preview-wrap {
      display: none;
      margin-top: 1rem;
      border-radius: var(--radius-sm);
      overflow: hidden;
      border: 1px solid var(--border);
      position: relative;
    }
    #preview-img {
      width: 100%;
      max-height: 320px;
      object-fit: contain;
      background: #111;
      display: block;
    }
    #clear-btn {
      position: absolute;
      top: 0.5rem;
      right: 0.5rem;
      background: rgba(0,0,0,0.65);
      border: none;
      color: var(--text);
      font-size: 1.1rem;
      cursor: pointer;
      border-radius: 50%;
      width: 2rem;
      height: 2rem;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: background var(--transition);
    }
    #clear-btn:hover { background: rgba(224,82,82,0.8); }

    /* ── Analyze button ── */
    #analyze-btn {
      width: 100%;
      margin-top: 1.2rem;
      padding: 0.85rem;
      border: none;
      border-radius: var(--radius-sm);
      background: var(--accent);
      color: #fff;
      font-size: 1rem;
      font-weight: 700;
      letter-spacing: 0.4px;
      cursor: pointer;
      transition: background var(--transition), transform var(--transition);
    }
    #analyze-btn:hover:not(:disabled) {
      background: var(--accent-h);
      transform: translateY(-1px);
    }
    #analyze-btn:disabled { opacity: 0.45; cursor: not-allowed; transform: none; }

    /* ── Spinner ── */
    .spinner-wrap {
      display: none;
      justify-content: center;
      align-items: center;
      gap: 0.8rem;
      padding: 1.4rem 0;
      color: var(--text-muted);
      font-size: 0.9rem;
    }
    .spinner {
      width: 22px; height: 22px;
      border: 3px solid var(--border);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.75s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── Result card ── */
    #result-card { display: none; }

    .verdict-text {
      font-size: clamp(1.6rem, 5vw, 2.2rem);
      font-weight: 800;
      letter-spacing: 1px;
      margin-bottom: 0.3rem;
    }
    .verdict-text.fake  { color: var(--red); }
    .verdict-text.real  { color: var(--green); }
    .verdict-text.noface { color: var(--grey); }

    .score-row {
      display: flex;
      align-items: center;
      gap: 0.8rem;
      margin-top: 1rem;
    }
    .score-label { font-size: 0.8rem; color: var(--text-muted); min-width: 5.5rem; }
    .bar-track {
      flex: 1;
      height: 10px;
      background: var(--surface2);
      border-radius: 999px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      border-radius: 999px;
      transition: width 0.5s ease;
      width: 0%;
    }
    .bar-fill.fake { background: var(--red); }
    .bar-fill.real { background: var(--green); }
    .bar-fill.noface { background: var(--grey); }
    .score-val { font-size: 0.85rem; font-weight: 600; min-width: 3.5rem; text-align: right; }

    .meta-row {
      margin-top: 1rem;
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }
    .chip {
      font-size: 0.75rem;
      padding: 0.25rem 0.65rem;
      border-radius: 999px;
      background: var(--surface2);
      border: 1px solid var(--border);
      color: var(--text-muted);
    }
    .chip strong { color: var(--text); }

    .threshold-note {
      margin-top: 0.8rem;
      font-size: 0.72rem;
      color: var(--text-muted);
    }

    /* ── Error ── */
    .error-msg {
      display: none;
      color: var(--red);
      font-size: 0.88rem;
      padding: 0.7rem 1rem;
      background: rgba(224,82,82,0.1);
      border: 1px solid rgba(224,82,82,0.3);
      border-radius: var(--radius-sm);
      margin-top: 0.8rem;
    }

    /* ── Footer ── */
    footer {
      margin-top: 2rem;
      font-size: 0.75rem;
      color: var(--text-muted);
      text-align: center;
    }
  </style>
</head>
<body>

  <header>
    <h1>Deep<span>Shield</span></h1>
    <p>Deepfake &amp; AI-Image Detector &mdash; Wavelet-CLIP + UniversalFakeDetect probe</p>
  </header>

  <!-- Model selector card -->
  <div class="card">
    <div class="label">Select Model</div>
    <div class="model-selector">
      <div class="model-btn active" data-model="1" id="btn-model-1">
        Model 1
        <span class="tag">face-swap &bull; recommended</span>
      </div>
      <div class="model-btn" data-model="2" id="btn-model-2">
        Model 2
        <span class="tag">face-swap &bull; old</span>
      </div>
      <div class="model-btn" data-model="3" id="btn-model-3">
        Model 3
        <span class="tag">AI-Gen &bull; StyleGAN/diffusion</span>
      </div>
    </div>
  </div>

  <!-- Upload card -->
  <div class="card">
    <div class="label">Upload Image</div>
    <div id="drop-zone">
      <div class="icon">&#128247;</div>
      <p><strong>Drag &amp; drop</strong> a photo here<br/>or <strong>click to browse</strong></p>
      <input type="file" id="file-input" accept="image/*" />
    </div>
    <div id="preview-wrap">
      <img id="preview-img" src="" alt="preview" />
      <button id="clear-btn" title="Remove image">&#x2715;</button>
    </div>

    <button id="analyze-btn" disabled>Analyze</button>

    <div class="spinner-wrap" id="spinner">
      <div class="spinner"></div>
      Running inference&hellip;
    </div>
    <div class="error-msg" id="error-msg"></div>
  </div>

  <!-- Result card -->
  <div class="card" id="result-card">
    <div class="label">Result</div>
    <div class="verdict-text" id="verdict-text">—</div>

    <div class="score-row">
      <span class="score-label" id="score-label">Score</span>
      <div class="bar-track">
        <div class="bar-fill" id="score-bar"></div>
      </div>
      <span class="score-val" id="score-val">—</span>
    </div>

    <div class="meta-row" id="meta-row"></div>
    <div class="threshold-note" id="threshold-note"></div>
  </div>

  <footer>DeepShield &bull; Models 1&amp;2: Wavelet-CLIP (thr 0.15) &bull; Model 3: CLIP-probe AI-Gen (thr 0.10)</footer>

  <script>
    // ── State ──────────────────────────────────────────────────────────────────
    let selectedModel = '1';
    let selectedFile  = null;

    // ── Model selector ─────────────────────────────────────────────────────────
    document.querySelectorAll('.model-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.model-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        selectedModel = btn.dataset.model;
      });
    });

    // ── Drop zone ──────────────────────────────────────────────────────────────
    const dropZone   = document.getElementById('drop-zone');
    const fileInput  = document.getElementById('file-input');
    const previewWrap = document.getElementById('preview-wrap');
    const previewImg = document.getElementById('preview-img');
    const analyzeBtn = document.getElementById('analyze-btn');
    const clearBtn   = document.getElementById('clear-btn');

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
      const f = e.dataTransfer.files[0];
      if (f && f.type.startsWith('image/')) loadFile(f);
    });
    fileInput.addEventListener('change', () => {
      if (fileInput.files[0]) loadFile(fileInput.files[0]);
    });
    clearBtn.addEventListener('click', () => {
      selectedFile = null;
      previewWrap.style.display = 'none';
      previewImg.src = '';
      analyzeBtn.disabled = true;
      fileInput.value = '';
      hideResult();
      hideError();
    });

    function loadFile(f) {
      selectedFile = f;
      const reader = new FileReader();
      reader.onload = ev => {
        previewImg.src = ev.target.result;
        previewWrap.style.display = 'block';
        analyzeBtn.disabled = false;
        hideResult();
        hideError();
      };
      reader.readAsDataURL(f);
    }

    // ── Analyze ────────────────────────────────────────────────────────────────
    analyzeBtn.addEventListener('click', async () => {
      if (!selectedFile) return;

      analyzeBtn.disabled = true;
      showSpinner(true);
      hideResult();
      hideError();

      const fd = new FormData();
      fd.append('image', selectedFile);
      fd.append('model_id', selectedModel);

      try {
        const resp = await fetch('/predict', { method: 'POST', body: fd });
        const data = await resp.json();
        if (!resp.ok || data.error) {
          showError(data.error || `Server error ${resp.status}`);
        } else {
          showResult(data);
        }
      } catch (err) {
        showError('Network error: ' + err.message);
      } finally {
        showSpinner(false);
        analyzeBtn.disabled = false;
      }
    });

    // ── UI helpers ─────────────────────────────────────────────────────────────
    function showSpinner(v) {
      document.getElementById('spinner').style.display = v ? 'flex' : 'none';
    }

    function showError(msg) {
      const el = document.getElementById('error-msg');
      el.textContent = msg;
      el.style.display = 'block';
    }

    function hideError() {
      document.getElementById('error-msg').style.display = 'none';
    }

    function hideResult() {
      document.getElementById('result-card').style.display = 'none';
    }

    function showResult(data) {
      const card       = document.getElementById('result-card');
      const verdictEl  = document.getElementById('verdict-text');
      const barEl      = document.getElementById('score-bar');
      const scoreValEl = document.getElementById('score-val');
      const scoreLbl   = document.getElementById('score-label');
      const metaRow    = document.getElementById('meta-row');
      const threshNote = document.getElementById('threshold-note');

      const isAigen  = data.model_id === '3';
      const verdict  = data.verdict;

      // Verdict text + color
      verdictEl.textContent = verdict;
      verdictEl.className   = 'verdict-text';
      if (isAigen) {
        if (verdict === 'AI-GENERATED')    verdictEl.classList.add('fake');
        else if (verdict === 'REAL')       verdictEl.classList.add('real');
        else                               verdictEl.classList.add('noface');
      } else {
        if (verdict.includes('DEEPFAKE'))  verdictEl.classList.add('fake');
        else if (verdict === 'REAL')       verdictEl.classList.add('real');
        else                               verdictEl.classList.add('noface');
      }

      // Score bar
      const score = isAigen ? data.aigen_score : data.wclip_score;
      scoreLbl.textContent = isAigen ? 'AI-Gen Score' : 'WCLIP Score';
      const cls = (verdict === 'AI-GENERATED' || verdict.includes('DEEPFAKE'))
                    ? 'fake'
                    : verdict === 'REAL' ? 'real' : 'noface';
      barEl.className = 'bar-fill ' + cls;

      if (score !== null && score !== undefined) {
        const pct = Math.min(Math.max(score * 100, 0), 100).toFixed(1);
        barEl.style.width    = pct + '%';
        scoreValEl.textContent = score.toFixed(4);
      } else {
        barEl.style.width    = '0%';
        scoreValEl.textContent = '—';
      }

      // Chips
      const modelLabels = {'1': 'Model 1 — face-swap', '2': 'Model 2 — face-swap (old)', '3': 'Model 3 — AI-Gen'};
      const modelLabel  = modelLabels[data.model_id] || ('Model ' + data.model_id);
      metaRow.innerHTML =
        '<span class="chip">Model: <strong>' + modelLabel + '</strong></span>' +
        (data.latency_ms !== undefined
          ? '<span class="chip">Latency: <strong>' + data.latency_ms + ' ms</strong></span>'
          : '');

      if (isAigen) {
        threshNote.textContent =
          'Detector: UniversalFakeDetect CLIP-probe (raw features, 768-d). ' +
          'Threshold: ' + data.threshold + ' — scores above classified as AI-GENERATED. ' +
          'Validation AUC=0.8956 on StyleGAN2 vs real photos.';
      } else {
        threshNote.textContent =
          'Decision threshold: ' + data.threshold +
          ' — scores above threshold are classified as FACE-SWAP DEEPFAKE.';
      }

      card.style.display = 'block';
    }
  </script>

</body>
</html>
"""

# ── Routes ───────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)


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
