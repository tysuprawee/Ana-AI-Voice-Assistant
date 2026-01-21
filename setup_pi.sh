#!/bin/bash

echo "🍓 Setting up Ana on Raspberry Pi..."

# Update System & Install FFMPEG
echo "📦 Installing System Dependencies..."
sudo apt-get update
sudo apt-get install -y ffmpeg git python3-pip

# Install Python Libraries (MINIMAL - no Whisper/PyTorch)
echo "🐍 Installing Python Libraries..."
pip3 install flask flask-cors --break-system-packages
pip3 install edge-tts --break-system-packages
pip3 install httpx ormsgpack --break-system-packages
pip3 install google-genai --break-system-packages
pip3 install tavily-python --break-system-packages

echo ""
echo "✅ Setup Complete!"
echo "🚀 Run: python3 server.py"
echo "🌐 Open Chromium: http://localhost:8080/avatar"
echo ""
echo "💡 Note: Speech recognition uses Chromium's built-in Web Speech API."
echo "   No Whisper or PyTorch needed! Much faster & lighter."
