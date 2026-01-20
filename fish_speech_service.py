#!/usr/bin/env python3
"""
Fish Speech TTS Service
=======================
A simple HTTP server that provides TTS with voice cloning using Fish Speech.
Run this separately from the main server (uses Python 3.11).
"""

import os
import io
import time
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# Fish Speech imports
import torch
import torchaudio
from fish_speech.models.vqgan.modules.firefly import FireflyArchitecture
from fish_speech.utils import autocast_exclude_mps

app = Flask(__name__)
CORS(app)

# Directories
VOICE_SAMPLES_DIR = Path("./voice_samples")
OUTPUTS_DIR = Path("./outputs")
VOICE_SAMPLES_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Model will be loaded on first request
model = None
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"🔧 Using device: {device}")


def load_model():
    """Load Fish Speech model (lazy loading)."""
    global model
    if model is not None:
        return model
    
    print("⏳ Loading Fish Speech model...")
    # This will download the model on first run
    try:
        from fish_speech.inference import load_model as fs_load_model
        model = fs_load_model("fishaudio/fish-speech-1.5", device=device)
        print("✅ Fish Speech model loaded!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise
    return model


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'device': device})


@app.route('/tts', methods=['POST'])
def text_to_speech():
    """Generate speech from text, optionally using a reference voice."""
    start_time = time.time()
    
    data = request.json
    text = data.get('text', '')
    voice_sample = data.get('voice_sample', None)  # Path to voice sample
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        # Load model if needed
        model = load_model()
        
        # Generate speech
        # TODO: Implement actual Fish Speech inference
        # For now, return a placeholder response
        
        latency = round((time.time() - start_time) * 1000)
        
        return jsonify({
            'success': True,
            'message': 'Fish Speech TTS service is ready',
            'latency_ms': latency
        })
        
    except Exception as e:
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
    
    # Save the voice sample
    filename = f"{voice_name}.wav"
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
    for f in VOICE_SAMPLES_DIR.glob('*.wav'):
        voices.append({
            'id': f.stem,
            'filename': f.name
        })
    for f in VOICE_SAMPLES_DIR.glob('*.mp3'):
        voices.append({
            'id': f.stem,
            'filename': f.name
        })
    
    return jsonify({'voices': voices})


if __name__ == '__main__':
    print("\n🐟 Fish Speech TTS Service")
    print("   Running on http://localhost:8081\n")
    app.run(debug=False, port=8081, host='0.0.0.0')
