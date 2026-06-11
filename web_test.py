"""
DeepShield — minimal stdlib web app to verify the models actually work.

No Streamlit, no Flask. Pure stdlib http.server, runs on the MAIN thread,
so it avoids the native torch teardown crash that kills the Streamlit app.

Run:
    python web_test.py
Then open http://localhost:8700 and upload an image.

Env overrides: SBI_CKPT / WCLIP_CKPT / WEIGHTS (same as app.py).
"""
import base64
import cgi
import io
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

sys.path.insert(0, str(Path(__file__).resolve().parent))

PORT = int(os.environ.get('PORT', '8700'))

# ── Load pipeline ONCE, on the main thread, before serving ──────────────────────
print("Loading DeepShield pipeline (this prints the model load below)...")
from inference.pipeline import DeepShieldPipeline

PIPELINE = DeepShieldPipeline(
    sbi_ckpt=os.environ.get('SBI_CKPT'),
    wclip_ckpt=os.environ.get('WCLIP_CKPT'),
    weights=os.environ.get('WEIGHTS'),
)
print(f"\n>>> Models ready. Open http://localhost:{PORT} in your browser.\n")


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>DeepShield test</title>
<style>
 body{{font-family:system-ui,sans-serif;max-width:760px;margin:40px auto;padding:0 16px;}}
 h1{{margin-bottom:0}} .sub{{color:#666;margin-top:4px}}
 .card{{border:1px solid #ddd;border-radius:10px;padding:20px;margin:18px 0}}
 .verdict{{font-size:1.6rem;font-weight:700;padding:6px 16px;border-radius:8px;color:#fff;display:inline-block}}
 .FAKE{{background:#ff4b4b}} .REAL{{background:#21c45e}} .UNKNOWN,.AI-GENERATED{{background:#f59e0b}}
 .bar{{background:#eee;border-radius:4px;height:14px;margin:4px 0}}
 .bar>div{{height:14px;border-radius:4px}}
 img{{max-width:340px;border-radius:8px}}
 pre{{background:#f6f6f6;padding:12px;border-radius:8px;overflow:auto;font-size:.8rem}}
 input[type=submit]{{padding:8px 20px;font-size:1rem;border-radius:6px;border:0;background:#2563eb;color:#fff;cursor:pointer}}
</style></head><body>
<h1>🛡️ DeepShield</h1>
<p class="sub">Model verification — SBI + Wavelet-CLIP ensemble</p>
<p>Ensemble weights: <b>SBI={w_sbi:.2f}</b> &nbsp; <b>WCLIP={w_wclip:.2f}</b> &nbsp;|&nbsp; device: <b>{device}</b></p>
<form method="post" enctype="multipart/form-data" class="card">
  <input type="file" name="image" accept="image/*" required>
  <input type="submit" value="Analyze">
</form>
{result}
</body></html>"""


def bar(score, label):
    pct = int(max(0.0, min(1.0, score)) * 100)
    color = '#ff4b4b' if score > 0.5 else '#21c45e'
    return (f'<div><b>{label}</b> — {pct}%'
            f'<div class="bar"><div style="width:{pct}%;background:{color}"></div></div></div>')


def render_result(result, img_b64):
    v = result.get('verdict', 'UNKNOWN')
    rows = [f'<div class="card"><span class="verdict {v}">{v}</span>'
            f' &nbsp; <small>latency {result.get("latency_ms", 0)} ms · track {result.get("track")}</small>']
    if img_b64:
        rows.append(f'<div style="margin:12px 0"><img src="data:image/jpeg;base64,{img_b64}"></div>')
    if result.get('track') == 1:
        rows.append(bar(result.get('confidence', 0), 'Overall (fake) confidence'))
        rows.append(bar(result.get('score_sbi', 0), 'SBI branch'))
        rows.append(bar(result.get('score_wclip', 0), 'Wavelet-CLIP branch'))
        hb = result.get('heatmap_b64')
        if hb:
            rows.append('<p><b>Grad-CAM:</b></p>'
                        f'<img src="data:image/png;base64,{hb}">')
    else:
        rows.append(bar(result.get('confidence', 0.5), 'AI-generated confidence'))
        rows.append('<p><small>No face detected → Track 2 (C2PA / SynthID metadata check)</small></p>')
    shown = {k: val for k, val in result.items() if k != 'heatmap_b64'}
    rows.append(f'<pre>{json.dumps(shown, indent=2)}</pre></div>')
    return '\n'.join(rows)


class Handler(BaseHTTPRequestHandler):
    def _page(self, result_html=''):
        html = PAGE.format(
            w_sbi=float(PIPELINE.ensemble.w_sbi),
            w_wclip=float(PIPELINE.ensemble.w_wclip),
            device=str(PIPELINE.device),
            result=result_html,
        )
        body = html.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith('/favicon'):
            self.send_response(204); self.end_headers(); return
        self._page()

    def do_POST(self):
        form = cgi.FieldStorage(
            fp=self.rfile, headers=self.headers,
            environ={'REQUEST_METHOD': 'POST',
                     'CONTENT_TYPE': self.headers['Content-Type']},
        )
        if 'image' not in form or not form['image'].file:
            self._page('<p style="color:red">No file uploaded.</p>'); return
        raw = form['image'].file.read()
        try:
            pil = Image.open(io.BytesIO(raw)).convert('RGB')
        except Exception as e:
            self._page(f'<p style="color:red">Bad image: {e}</p>'); return

        img_rgb = np.array(pil)
        try:
            result = PIPELINE.detect_array(img_rgb)
        except Exception as e:
            import traceback
            self._page(f'<pre style="color:red">{traceback.format_exc()}</pre>'); return

        buf = io.BytesIO(); pil.save(buf, format='JPEG')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        self._page(render_result(result, img_b64))

    def log_message(self, fmt, *args):
        print("  [http]", fmt % args)


if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', PORT), Handler)
    print(f">>> Serving on http://localhost:{PORT}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
