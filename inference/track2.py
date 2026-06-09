"""
Track 2: AI-generated image detection (no face detected case).
Checks C2PA metadata and Google SynthID API.

Both checks are optional — set environment variables to enable:
  SYNTHID_API_KEY   - Google SynthID Detector API key
  SYNTHID_PROJECT   - Google Cloud project ID

C2PA requires: pip install c2pa-python (bundled in requirements.txt)
If c2pa-python unavailable, that check is skipped gracefully.
"""
import hashlib
import json
import os

c2pa = None
_C2PA = False
try:
    import c2pa  # type: ignore[import-untyped,assignment]
    _C2PA = True
except ImportError:
    pass


def check_c2pa(image_path: str) -> dict:
    """
    Read C2PA provenance metadata from image.
    Returns dict with 'is_ai', 'has_manifest', and 'generator' fields.
    """
    result = {'is_ai': False, 'has_manifest': False, 'generator': None, 'error': None}

    if not _C2PA:
        result['error'] = 'c2pa-python not installed'
        return result

    try:
        manifest_store = c2pa.read_file(image_path, None)  # type: ignore[union-attr]
        if manifest_store is None:
            return result

        data = json.loads(manifest_store)
        result['has_manifest'] = True

        # Check active manifest for AI generation claims
        active_id = data.get('active_manifest')
        if active_id and active_id in data.get('manifests', {}):
            manifest = data['manifests'][active_id]
            assertions = manifest.get('assertions', [])
            for assertion in assertions:
                label = assertion.get('label', '').lower()
                if 'ai' in label or 'generative' in label or 'created' in label:
                    result['is_ai'] = True
                    result['generator'] = assertion.get('data', {}).get('generator', 'unknown')
                    break

    except Exception as e:
        result['error'] = str(e)

    return result


def check_synthid(image_path: str) -> dict:
    """
    Check for Google SynthID watermark.
    Requires SYNTHID_API_KEY and SYNTHID_PROJECT env vars.
    Returns dict with 'detected', 'confidence', 'error'.
    """
    result = {'detected': False, 'confidence': None, 'error': None}

    api_key = os.environ.get('SYNTHID_API_KEY')
    project = os.environ.get('SYNTHID_PROJECT')

    if not api_key or not project:
        result['error'] = 'SYNTHID_API_KEY or SYNTHID_PROJECT not set'
        return result

    try:
        import requests  # type: ignore[import-untyped]
        with open(image_path, 'rb') as f:
            image_data = f.read()

        import base64
        b64_image = base64.b64encode(image_data).decode()

        url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{project}/locations/us-central1:detectWatermark"
        headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
        payload = {'instances': [{'b64': b64_image}]}

        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        predictions = data.get('predictions', [{}])
        if predictions:
            conf = predictions[0].get('watermark_probability', 0.0)
            result['detected']   = conf > 0.5
            result['confidence'] = round(float(conf), 4)

    except Exception as e:
        result['error'] = str(e)

    return result


def compute_evidence_hash(detection_id: str, verdict: str, confidence: float, timestamp: str) -> str:
    """SHA-256 hash of detection result for audit trail."""
    data = f"{detection_id}|{verdict}|{confidence:.6f}|{timestamp}"
    return hashlib.sha256(data.encode()).hexdigest()


def track2_detect(image_path: str) -> dict:
    """
    Full Track 2 pipeline: C2PA metadata + SynthID watermark check.
    Called when no face is detected in the image.
    """
    c2pa_result    = check_c2pa(image_path)
    synthid_result = check_synthid(image_path)

    is_ai = c2pa_result['is_ai'] or synthid_result['detected']

    if is_ai:
        verdict = 'AI-GENERATED'
        confidence = 0.9
        if synthid_result['confidence'] is not None:
            confidence = synthid_result['confidence']
    elif c2pa_result['error'] and synthid_result['error']:
        verdict = 'UNKNOWN'
        confidence = 0.5
    else:
        verdict = 'REAL'
        confidence = 0.1

    return {
        'verdict':    verdict,
        'confidence': confidence,
        'track':      2,
        'c2pa':       c2pa_result,
        'synthid':    synthid_result,
    }
