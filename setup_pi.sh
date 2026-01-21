#!/bin/bash

echo "🍓 Setting up Ana on Raspberry Pi..."

# Update System & Install FFMPEG
echo "📦 Installing System Dependencies..."
sudo apt-get update
sudo apt-get install -y ffmpeg git python3-pip unzip wget

# Install Python Libraries
echo "🐍 Installing Python Libraries..."
pip3 install --user flask flask-cors
pip3 install --user edge-tts
pip3 install --user httpx ormsgpack
pip3 install --user google-genai
pip3 install --user tavily-python
pip3 install --user vosk

# Download Vosk Model
echo "📥 Downloading Vosk Speech Recognition Model..."
if [ ! -d "vosk-model-small-en-us-0.15" ]; then
    wget -q --show-progress https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
    unzip -q vosk-model-small-en-us-0.15.zip
    rm vosk-model-small-en-us-0.15.zip
    echo "✅ Vosk model downloaded!"
else
    echo "✅ Vosk model already exists"
fi

echo ""
echo "✅ Setup Complete!"
echo ""
echo "🚀 Run: python3 server.py"
echo "🌐 Open Chromium: http://localhost:8080/avatar"
echo ""
echo "💡 Speech recognition is now OFFLINE using Vosk!"
echo "   No internet needed for voice commands."
