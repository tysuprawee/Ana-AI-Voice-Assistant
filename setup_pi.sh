#!/bin/bash

echo "🍓 Setting up Ana on Raspberry Pi..."

# Update System & Install FMPEG (Critical for Audio)
echo "📦 Installing System Dependencies (ffmpeg, portaudio)..."
sudo apt-get update
sudo apt-get install -y ffmpeg portaudio19-dev python3-pyaudio git python3-pip

# Install Python Libraries
echo "🐍 Installing Python Libraries..."
pip3 install -r requirements.txt

echo "✅ Setup Complete!"
echo "🚀 Run: python3 server.py"
echo "🌐 Open: http://localhost:8080/avatar"
