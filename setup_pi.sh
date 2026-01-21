#!/bin/bash

echo "🍓 Setting up Ana on Raspberry Pi..."

# Update System & Install FFMPEG
echo "📦 Installing System Dependencies..."
sudo apt-get update
sudo apt-get install -y ffmpeg git python3-pip

# Install Python Libraries (user mode for older pip)
echo "🐍 Installing Python Libraries..."
pip3 install --user flask flask-cors
pip3 install --user edge-tts
pip3 install --user httpx ormsgpack
pip3 install --user google-genai
pip3 install --user tavily-python

echo ""
echo "✅ Setup Complete!"
echo "🚀 Run: python3 server.py"
echo "🌐 Open Chromium: http://localhost:8080/avatar"
echo ""
echo "💡 Note: Speech recognition uses Chromium's built-in Web Speech API."
echo "   No Whisper or PyTorch needed! Much faster & lighter."
