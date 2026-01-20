#!/usr/bin/env python3
"""
Speech-to-Text using OpenAI Whisper
====================================
100% free, runs locally, no API key needed!

Usage:
    python stt_whisper.py audio.mp3
    python stt_whisper.py audio.mp3 --model base
    python stt_whisper.py audio.mp3 --language th
"""

import argparse
import sys
import os
from pathlib import Path

try:
    import whisper
except ImportError:
    print("❌ Whisper not installed. Install with: pip install openai-whisper")
    sys.exit(1)


# Model sizes and their requirements
MODELS = {
    "tiny":   {"size": "~39 MB",  "vram": "~1 GB",  "speed": "~10x", "quality": "★☆☆☆☆"},
    "base":   {"size": "~74 MB",  "vram": "~1 GB",  "speed": "~7x",  "quality": "★★☆☆☆"},
    "small":  {"size": "~244 MB", "vram": "~2 GB",  "speed": "~4x",  "quality": "★★★☆☆"},
    "medium": {"size": "~769 MB", "vram": "~5 GB",  "speed": "~2x",  "quality": "★★★★☆"},
    "large":  {"size": "~1550 MB","vram": "~10 GB", "speed": "~1x",  "quality": "★★★★★"},
}


def list_models():
    """Display available Whisper models."""
    print("\n🎙️  Available Whisper Models")
    print("=" * 60)
    print(f"{'Model':<10} {'Size':<12} {'VRAM':<10} {'Speed':<8} {'Quality'}")
    print("-" * 60)
    for name, info in MODELS.items():
        print(f"{name:<10} {info['size']:<12} {info['vram']:<10} {info['speed']:<8} {info['quality']}")
    print("\n💡 Tip: Start with 'base' for good balance of speed and accuracy")
    print("   Use 'tiny' for fastest results, 'large' for best accuracy")


def transcribe_audio(
    audio_path: str,
    model_name: str = "base",
    language: str = None,
    task: str = "transcribe",
    output_format: str = "text",
    output_file: str = None,
):
    """Transcribe audio to text using Whisper."""
    
    # Validate file exists
    if not os.path.exists(audio_path):
        print(f"❌ File not found: {audio_path}")
        sys.exit(1)
    
    print(f"\n🎙️  Whisper Speech-to-Text")
    print("=" * 50)
    print(f"📁 Audio: {audio_path}")
    print(f"🧠 Model: {model_name}")
    if language:
        print(f"🌍 Language: {language}")
    print(f"📋 Task: {task}")
    print("-" * 50)
    
    # Load model
    print(f"\n⏳ Loading {model_name} model...")
    try:
        model = whisper.load_model(model_name)
        print("✅ Model loaded!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Transcribe
    print("\n⏳ Transcribing audio...")
    try:
        options = {
            "task": task,
            "verbose": False,
        }
        if language:
            options["language"] = language
        
        result = model.transcribe(audio_path, **options)
        
    except Exception as e:
        print(f"❌ Error transcribing: {e}")
        sys.exit(1)
    
    # Output results
    text = result["text"].strip()
    detected_lang = result.get("language", "unknown")
    
    print("\n" + "=" * 50)
    print("📝 TRANSCRIPTION RESULT")
    print("=" * 50)
    print(f"\n🌍 Detected Language: {detected_lang}")
    print(f"\n📄 Text:\n")
    print(text)
    print("\n" + "=" * 50)
    
    # Save to file if requested
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n💾 Saved to: {output_file}")
    
    # Show segments if verbose
    if output_format == "detailed":
        print("\n📊 Segments:")
        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            seg_text = segment["text"]
            print(f"  [{start:.2f}s - {end:.2f}s] {seg_text}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="🎙️ Whisper STT - Free Speech-to-Text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stt_whisper.py audio.mp3
  python stt_whisper.py audio.mp3 --model small
  python stt_whisper.py thai_audio.mp3 --language th
  python stt_whisper.py audio.mp3 --translate
  python stt_whisper.py --list-models
        """
    )
    
    parser.add_argument(
        "audio",
        nargs="?",
        help="Audio file to transcribe (mp3, wav, m4a, etc.)"
    )
    
    parser.add_argument(
        "--model", "-m",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Model size (default: base)"
    )
    
    parser.add_argument(
        "--language", "-l",
        help="Language code (e.g., 'en', 'th', 'ja'). Auto-detected if not specified."
    )
    
    parser.add_argument(
        "--translate", "-t",
        action="store_true",
        help="Translate to English instead of transcribe"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Save transcription to file"
    )
    
    parser.add_argument(
        "--detailed", "-d",
        action="store_true",
        help="Show detailed output with timestamps"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models"
    )
    
    args = parser.parse_args()
    
    if args.list_models:
        list_models()
        return
    
    if not args.audio:
        parser.print_help()
        print("\n💡 Quick start: python stt_whisper.py audio.mp3")
        return
    
    task = "translate" if args.translate else "transcribe"
    output_format = "detailed" if args.detailed else "text"
    
    transcribe_audio(
        args.audio,
        args.model,
        args.language,
        task,
        output_format,
        args.output,
    )


if __name__ == "__main__":
    main()
