#!/usr/bin/env python3
"""
Fish Speech TTS Service
=======================
HTTP server providing TTS with voice cloning using Fish Speech.
Run with Python 3.11: source fishspeech_venv/bin/activate && python fish_tts_server.py
"""

import os
import io
import time
import wave
import tempfile
from pathlib import Path

import torch
import numpy as np

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Directories
VOICE_SAMPLES_DIR = Path("./voice_samples")
OUTPUTS_DIR = Path("./outputs")
VOICE_SAMPLES_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🔧 Using device: {device}")

# Lazy load engine
_engine = None


def get_engine():
    """Get or create the TTS inference engine."""
    global _engine
    if _engine is None:
        print("⏳ Loading Fish Speech model (first run may take a while)...")
        from fish_speech.inference_engine import TTSInferenceEngine
        _engine = TTSInferenceEngine()
        print("✅ Fish Speech model loaded!")
    return _engine


@app.route('/health', methods=['GET'])
def health():
    """Health check."""
    return jsonify({
        'status': 'ok',
        'service': 'Fish Speech TTS',
        'device': device
    })


@app.route('/tts', methods=['POST'])
def text_to_speech():
    """Generate speech from text with optional voice reference."""
    start_time = time.time()
    
    data = request.json or {}
    text = data.get('text', '')
    voice_id = data.get('voice_id', None)  # Name of voice sample
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        engine = get_engine()
        
        # Find voice reference if specified
        reference_audio = None
        if voice_id:
            for ext in ['.wav', '.mp3', '.m4a']:
                ref_path = VOICE_SAMPLES_DIR / f"{voice_id}{ext}"
                if ref_path.exists():
                    reference_audio = str(ref_path)
                    break
        
        # Generate speech
        # Fish Speech API may vary - adjust based on actual API
        audio_data = engine.tts(
            text=text,
            reference_audio=reference_audio,
        )
        
        # Save to file
        filename = f"fish_{int(time.time() * 1000)}.wav"
        output_path = OUTPUTS_DIR / filename
        
        # Save audio (assuming audio_data is numpy array or tensor)
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.cpu().numpy()
        
        # Write WAV file
        import soundfile as sf
        sf.write(str(output_path), audio_data, 44100)
        
        latency = round((time.time() - start_time) * 1000)
        
        return jsonify({
            'success': True,
            'audio_url': f'/outputs/{filename}',
            'latency_ms': latency,
            'voice_id': voice_id or 'default'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/clone', methods=['POST'])
def clone_voice():
    """Upload a voice sample for cloning."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    voice_name = request.form.get('name', 'my_voice')
    
    # Clean the name
    voice_name = "".join(c for c in voice_name if c.isalnum() or c in '_-')
    
    # Get original extension
    original_ext = Path(audio_file.filename).suffix or '.wav'
    
    # Save the voice sample
    filename = f"{voice_name}{original_ext}"
    filepath = VOICE_SAMPLES_DIR / filename
    audio_file.save(str(filepath))
    
    return jsonify({
        'success': True,
        'message': f'Voice sample saved as {filename}',
        'voice_id': voice_name
    })


@app.route('/voices', methods=['GET'])
def list_voices():
    """List available cloned voices."""
    voices = []
    for ext in ['*.wav', '*.mp3', '*.m4a']:
        for f in VOICE_SAMPLES_DIR.glob(ext):
            voices.append({
                'id': f.stem,
                'filename': f.name,
                'size_bytes': f.stat().st_size
            })
    
    return jsonify({
        'voices': voices,
        'count': len(voices)
    })


@app.route('/outputs/<path:filename>')
def serve_output(filename):
    """Serve generated audio files."""
    return send_from_directory(OUTPUTS_DIR, filename)


if __name__ == '__main__':
    print("\n🐟 Fish Speech TTS Service")
    print("   http://localhost:8081")
    print("   Endpoints:")
    print("   - POST /tts - Generate speech")
    print("   - POST /clone - Upload voice sample")
    print("   - GET /voices - List cloned voices")
    print()
    
    # Pre-load the model
    get_engine()
    
    app.run(debug=False, port=8081, host='0.0.0.0')
