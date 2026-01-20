# 🍓 Ana: Raspberry Pi AI Assistant

This guide explains how to run Ana on your Raspberry Pi 4.

## 1. Transfer Files
Copy this entire folder (`TTS`) to your Raspberry Pi.
You can use a USB drive, `scp`, or drag-and-drop.

## 2. Setup
Open a terminal inside the folder and run:

```bash
sh setup_pi.sh
```

This will satisfy all dependencies (ffmpeg, python libraries).

## 3. Run
Start the server:

```bash
python3 server.py
```

## 4. Usage
Open Chromium Browser on the Pi and go to:
**http://localhost:8080/avatar**

**Tips:**
*   **Full Screen:** Press F11.
*   **Microphone:** Ensure your USB mic is selected in Pi settings if she creates empty recordings.
*   **Speed:** If she is slow to listen, we can switch to the 'Tiny' Whisper model later.
