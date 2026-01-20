#!/usr/bin/env python3
"""
Edge TTS API Tester
===================
Uses Microsoft Edge's free TTS service with high-quality voices.
No API key required!

Usage:
    python tts_edge.py "Hello, world!"
    python tts_edge.py "Hello!" --voice en-US-AriaNeural --output hello.mp3
    python tts_edge.py --list-voices
    python tts_edge.py --list-voices --language en
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

try:
    import edge_tts
except ImportError:
    print("❌ edge-tts not installed. Install with: pip install edge-tts")
    sys.exit(1)


# Popular high-quality voices
POPULAR_VOICES = {
    "en-US": [
        ("en-US-AriaNeural", "Aria", "Female - Conversational, friendly"),
        ("en-US-JennyNeural", "Jenny", "Female - Clear, professional"),
        ("en-US-GuyNeural", "Guy", "Male - Warm, friendly"),
        ("en-US-ChristopherNeural", "Christopher", "Male - Professional"),
        ("en-US-EricNeural", "Eric", "Male - Casual"),
        ("en-US-MichelleNeural", "Michelle", "Female - Warm"),
    ],
    "en-GB": [
        ("en-GB-SoniaNeural", "Sonia", "Female - British"),
        ("en-GB-RyanNeural", "Ryan", "Male - British"),
    ],
    "ja-JP": [
        ("ja-JP-NanamiNeural", "Nanami", "Female - Japanese"),
        ("ja-JP-KeitaNeural", "Keita", "Male - Japanese"),
    ],
    "ko-KR": [
        ("ko-KR-SunHiNeural", "SunHi", "Female - Korean"),
        ("ko-KR-InJoonNeural", "InJoon", "Male - Korean"),
    ],
    "zh-CN": [
        ("zh-CN-XiaoxiaoNeural", "Xiaoxiao", "Female - Chinese"),
        ("zh-CN-YunxiNeural", "Yunxi", "Male - Chinese"),
    ],
    "th-TH": [
        ("th-TH-PremwadeeNeural", "Premwadee", "Female - Thai"),
        ("th-TH-NiwatNeural", "Niwat", "Male - Thai"),
    ],
}


async def list_voices(language_filter: str = None):
    """List all available voices, optionally filtered by language."""
    print("\n🎙️  Available Edge TTS Voices")
    print("=" * 60)
    
    voices = await edge_tts.list_voices()
    
    # Group by language
    by_language = {}
    for voice in voices:
        lang = voice["Locale"]
        if language_filter and not lang.lower().startswith(language_filter.lower()):
            continue
        if lang not in by_language:
            by_language[lang] = []
        by_language[lang].append(voice)
    
    for lang in sorted(by_language.keys()):
        print(f"\n📍 {lang}")
        print("-" * 40)
        for voice in by_language[lang]:
            gender_icon = "👩" if voice["Gender"] == "Female" else "👨"
            print(f"  {gender_icon} {voice['ShortName']}")
            print(f"     └─ {voice['FriendlyName']}")
    
    print(f"\n✅ Total: {sum(len(v) for v in by_language.values())} voices")
    
    if not language_filter:
        print("\n💡 Tip: Use --language en to filter by language code")


async def text_to_speech(
    text: str,
    voice: str = "en-US-AriaNeural",
    output_file: str = "output.mp3",
    rate: str = "+0%",
    pitch: str = "+0Hz",
    volume: str = "+0%",
):
    """Convert text to speech and save as MP3."""
    print(f"\n🎤 Edge TTS - Text to Speech")
    print("=" * 50)
    print(f"📝 Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"🎙️  Voice: {voice}")
    print(f"⚡ Rate: {rate}")
    print(f"🎵 Pitch: {pitch}")
    print(f"🔊 Volume: {volume}")
    print(f"💾 Output: {output_file}")
    print("-" * 50)
    
    try:
        communicate = edge_tts.Communicate(
            text,
            voice,
            rate=rate,
            pitch=pitch,
            volume=volume,
        )
        
        await communicate.save(output_file)
        
        file_size = os.path.getsize(output_file)
        print(f"\n✅ Success! Saved to: {output_file}")
        print(f"📦 File size: {file_size / 1024:.1f} KB")
        
        # Try to play the audio
        await play_audio(output_file)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


async def play_audio(file_path: str):
    """Try to play the audio file using available system tools."""
    import subprocess
    import platform
    
    system = platform.system()
    
    print(f"\n🔊 Playing audio...")
    
    try:
        if system == "Darwin":  # macOS
            subprocess.run(["afplay", file_path], check=True)
        elif system == "Windows":
            # Use Windows Media Player
            os.startfile(file_path)
        elif system == "Linux":
            # Try common Linux audio players
            for player in ["mpv", "ffplay", "aplay", "paplay"]:
                try:
                    subprocess.run([player, file_path], check=True, 
                                   stdout=subprocess.DEVNULL, 
                                   stderr=subprocess.DEVNULL)
                    break
                except FileNotFoundError:
                    continue
        print("✅ Playback complete!")
    except Exception as e:
        print(f"⚠️  Could not auto-play. Open {file_path} manually.")


async def interactive_mode():
    """Run in interactive mode for testing multiple phrases."""
    print("\n🎙️  Edge TTS Interactive Mode")
    print("=" * 50)
    print("Type text to convert to speech. Commands:")
    print("  /voice <name>  - Change voice")
    print("  /voices        - List popular voices")
    print("  /quit          - Exit")
    print("=" * 50)
    
    voice = "en-US-AriaNeural"
    
    while True:
        try:
            text = input(f"\n[{voice}] > ").strip()
            
            if not text:
                continue
            
            if text.lower() == "/quit":
                print("👋 Goodbye!")
                break
            
            if text.lower() == "/voices":
                print("\n🌟 Popular Voices:")
                for lang, voices in POPULAR_VOICES.items():
                    print(f"\n  {lang}:")
                    for v_id, v_name, v_desc in voices:
                        print(f"    • {v_id} ({v_desc})")
                continue
            
            if text.lower().startswith("/voice "):
                voice = text.split(" ", 1)[1].strip()
                print(f"✅ Voice changed to: {voice}")
                continue
            
            # Generate speech
            output_file = "interactive_output.mp3"
            await text_to_speech(text, voice, output_file)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            break


def main():
    parser = argparse.ArgumentParser(
        description="🎙️ Edge TTS API Tester - Free high-quality text-to-speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tts_edge.py "Hello, world!"
  python tts_edge.py "Hello!" --voice en-US-JennyNeural
  python tts_edge.py "Fast speech" --rate "+50%"
  python tts_edge.py --list-voices --language ja
  python tts_edge.py --interactive
        """
    )
    
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to convert to speech"
    )
    
    parser.add_argument(
        "--voice", "-v",
        default="en-US-AriaNeural",
        help="Voice to use (default: en-US-AriaNeural)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="output.mp3",
        help="Output file path (default: output.mp3)"
    )
    
    parser.add_argument(
        "--rate", "-r",
        default="+0%",
        help="Speech rate, e.g., '+50%%' or '-25%%' (default: +0%%)"
    )
    
    parser.add_argument(
        "--pitch", "-p",
        default="+0Hz",
        help="Voice pitch, e.g., '+10Hz' or '-5Hz' (default: +0Hz)"
    )
    
    parser.add_argument(
        "--volume",
        default="+0%",
        help="Volume adjustment, e.g., '+20%%' (default: +0%%)"
    )
    
    parser.add_argument(
        "--list-voices", "-l",
        action="store_true",
        help="List all available voices"
    )
    
    parser.add_argument(
        "--language",
        help="Filter voices by language code (e.g., 'en', 'ja', 'ko')"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Run appropriate mode
    if args.list_voices:
        asyncio.run(list_voices(args.language))
    elif args.interactive:
        asyncio.run(interactive_mode())
    elif args.text:
        asyncio.run(text_to_speech(
            args.text,
            args.voice,
            args.output,
            args.rate,
            args.pitch,
            args.volume,
        ))
    else:
        parser.print_help()
        print("\n💡 Quick start: python tts_edge.py \"Hello, world!\"")


if __name__ == "__main__":
    main()
