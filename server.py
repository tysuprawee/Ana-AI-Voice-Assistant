#!/usr/bin/env python3
"""
TTS + STT + AI API Server
=========================
Flask server with edge-tts, whisper, and Gemini AI.
"""

import os
import time
import asyncio
import tempfile
import re
from pathlib import Path

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

import edge_tts
import httpx
import ormsgpack
from google import genai
from tavily import TavilyClient

# Whisper is OPTIONAL (not needed on Raspberry Pi)
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️  Whisper not installed. /api/voice-chat will be unavailable.")
    print("   Use /api/text-chat instead (browser handles speech recognition).")

app = Flask(__name__, static_folder='.')
CORS(app)

# Gemini API Key
GEMINI_API_KEY = "AIzaSyBFUEN4Md2lGc97CzlW0deS7jt-ftdw5l0"

# Tavily API Key (for real-time search)
TAVILY_API_KEY = "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"

# Fish Audio API Key (for voice cloning)
FISH_AUDIO_API_KEY = "2c97ce52e63e43ddb7beb5a0c61564d3"
FISH_AUDIO_BASE_URL = "https://api.fish.audio"

# Initialize clients
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

GOOGLE_CLOUD_API_KEY = "AIzaSyBVgQFHn0U1Lya7J-UsxOW5ll4NgCl-WAU"

# Voice samples directory
VOICE_SAMPLES_DIR = Path("./voice_samples")
VOICE_SAMPLES_DIR.mkdir(exist_ok=True)

# Load Whisper model IF available
whisper_model = None
if WHISPER_AVAILABLE:
    print("⏳ Loading Whisper model...")
    whisper_model = whisper.load_model("base")
    print("✅ Whisper model loaded!")

# Output directory
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def cleanup_outputs():
    """Delete old audio files to save space on Pi."""
    try:
        # Keep max 50 files or delete older than 10 mins
        files = sorted(OUTPUT_DIR.glob("*.mp3"), key=lambda f: f.stat().st_mtime)
        
        # 1. Cap at 50 files
        while len(files) > 50:
            files[0].unlink()
            files.pop(0)
            
        # 2. Delete older than 10 mins
        now = time.time()
        for f in files:
            if now - f.stat().st_mtime > 600: # 600s = 10 mins
                f.unlink()
    except Exception as e:
        print(f"Cleanup error: {e}")

# Keywords that trigger a web search
SEARCH_TRIGGERS = [
    # English time / recency
    "news", "latest", "today", "yesterday", "recent", "currently",
    "right now", "this week", "this month", "just happened",

    # Results / events
    "what happened", "who won", "score", "scores", "result", "results",

    # Live-ish info
    "weather", "forecast",
    "price", "prices", "stock", "stocks", "bitcoin", "btc", "crypto",
    "exchange rate", "fx rate",

    # Politics / elections
    "election", "elections", "president", "prime minister",

    # Thai recency
    "ข่าว", "วันนี้", "ตอนนี้", "เมื่อวาน", "ล่าสุด",
    "เมื่อกี้", "เมื่อครู่", "เมื่อไม่กี่วัน",  # common “recently” phrases

    # Thai live-ish info
    "ราคา", "ราคาตอนนี้", "ราคาเท่าไหร่",
    "อากาศ", "พยากรณ์อากาศ",
    "ผลบอล", "ผลแข่ง", "ผลการแข่งขัน",

    # Thai finance / crypto
    "บิทคอยน์", "บิตคอยน์", "หุ้น", "ค่าเงิน", "อัตราแลกเปลี่ยน"
]

def needs_search(text: str) -> bool:
    """Check if the query needs real-time search."""
    text_lower = text.lower()
    return any(trigger in text_lower for trigger in SEARCH_TRIGGERS)

def search_web(query: str, max_results: int = 3) -> str:
    """Search the web using Tavily and return context."""
    try:
        response = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=True
        )
        
        # Build context from search results
        context_parts = []
        
        # Add Tavily's AI answer if available
        if response.get('answer'):
            context_parts.append(f"Summary: {response['answer']}")
        
        # Add top results
        for result in response.get('results', [])[:max_results]:
            title = result.get('title', '')
            content = result.get('content', '')[:200]
            context_parts.append(f"- {title}: {content}")
        
        return "\n".join(context_parts) if context_parts else ""
    except Exception as e:
        print(f"Search error: {e}")
        return ""


@app.route('/')
def index():
    return send_from_directory('.', 'ui.html')


@app.route('/dashboard')
def voice_dashboard():
    return send_from_directory('.', 'voice_dashboard.html')


@app.route('/tts')
def tts_page():
    return send_from_directory('.', 'tts.html')


@app.route('/avatar')
def avatar_page():
    return send_from_directory('.', 'avatar.html')


@app.route('/api/tts', methods=['POST'])
def text_to_speech():
    """Convert text to speech using Edge-TTS or Fish Audio."""
    start_time = time.time()
    
    data = request.json
    text = data.get('text', '')
    voice = data.get('voice', 'en-US-AriaNeural')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Check if using Fish Audio (voice starts with 'fish:')
    # Check if using Fish Audio (voice starts with 'fish:')
    if voice.startswith('fish:'):
        # Use Fish Audio API
        reference_id = voice.replace('fish:', '')
        return fish_audio_tts(text, reference_id, start_time)
    # Check if using Google Cloud TTS (voice starts with 'googlecloud:')
    elif voice.startswith('googlecloud:'):
        voice_id = voice.replace('googlecloud:', '')
        return google_cloud_tts_generate(text, voice_id, start_time)
    # Check if using Google TTS (voice starts with 'google:')
    elif voice.startswith('google:'):
        lang = voice.replace('google:', '')
        return google_tts_generate(text, lang, start_time)
    else:
        # Use Edge-TTS with prosody settings
        rate = data.get('rate', '+0%')  # e.g., '-10%', '+20%', 'slow', 'fast'
        pitch = data.get('pitch', '+0Hz')  # e.g., '+10Hz', '-5Hz'
        style = data.get('style', None)  # e.g., 'cheerful', 'friendly'
        pauses = data.get('pauses', False)  # Add natural pauses
        warmth = data.get('warmth', False)  # Add audio warmth processing
        return edge_tts_generate(text, voice, start_time, rate, pitch, style, pauses, warmth)


def edge_tts_generate(text: str, voice: str, start_time: float, 
                      rate: str = '+0%', pitch: str = '+0Hz', style: str = None,
                      pauses: bool = False, warmth: bool = False):
    """Generate speech using Edge-TTS with prosody controls."""
    filename = f"tts_{int(time.time() * 1000)}.mp3"
    output_path = OUTPUT_DIR / filename
    
    # Add natural pauses by inserting ellipsis markers
    if pauses:
        import re
        
        # Check if text contains Thai characters
        has_thai = bool(re.search(r'[\u0E00-\u0E7F]', text))
        
        if has_thai:
            # Use PyThaiNLP for smart Thai word segmentation
            try:
                from pythainlp import word_tokenize
                
                # Tokenize Thai text into words/phrases, keeping whitespace as tokens
                words = word_tokenize(text, engine='newmm', keep_whitespace=True)
                
                # Build text with natural pauses based on USER INPUT only
                result = []
                for i, word in enumerate(words):
                    # Check if the word is whitespace (space, tab, newline)
                    if word.strip() == '':
                        # Convert user's space/newline to explicit pause
                        result.append(' ... ')
                    else:
                        result.append(word)
                        # No auto-pauses added! Strict adherence to user spacing.
                
                text = ''.join(result)
                
                text = ''.join(result)
                
            except Exception as e:
                # Fallback to simple space-based pauses
                print(f"PyThaiNLP error: {e}")
                # Replace multiple spaces with a single pause
                text = re.sub(r'\s+', ' ... ', text)
        else:
            # For English/other languages, pause after punctuation
            text = re.sub(r'"', '" ... ', text)
            text = re.sub(r'"', '" ... ', text)
            text = re.sub(r'\.\s+', '. ... ', text)
            text = re.sub(r'\?\s+', '? ... ', text)
            text = re.sub(r'!\s+', '! ... ', text)
    
    async def generate():
        # Edge-TTS Communicate accepts rate and pitch directly
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        await communicate.save(str(output_path))
    
    asyncio.run(generate())
    
    # Apply audio post-processing for warmth
    if warmth:
        try:
            from pydub import AudioSegment
            from pydub.effects import normalize
            
            # Load the generated audio
            audio = AudioSegment.from_mp3(str(output_path))
            
            # Apply warmth effects:
            # 1. Slight bass boost (low-pass filter effect simulation)
            audio = audio.low_pass_filter(8000)  # Soften high frequencies
            
            # 2. Add subtle warmth by boosting low frequencies
            # Split into bands and boost bass slightly
            low = audio.low_pass_filter(300)
            high = audio.high_pass_filter(300)
            audio = low + 3 + high  # Boost bass by 3dB
            
            # 3. Normalize audio levels
            audio = normalize(audio)
            
            # 4. Export with slightly higher quality
            audio.export(str(output_path), format="mp3", bitrate="192k")
            
        except Exception as e:
            print(f"Audio processing warning: {e}")
            # Continue with unprocessed audio
    
    latency = round((time.time() - start_time) * 1000)
    
    return jsonify({
        'success': True,
        'audio_url': f'/outputs/{filename}',
        'latency_ms': latency,
        'voice': voice,
        'provider': 'edge-tts',
        'text_length': len(text),
        'settings': {'rate': rate, 'pitch': pitch, 'style': style, 'pauses': pauses, 'warmth': warmth}
    })



def google_tts_generate(text: str, lang: str, start_time: float):
    """Generate speech using Google TTS (gTTS)."""
    try:
        from gtts import gTTS
        
        filename = f"google_{int(time.time() * 1000)}.mp3"
        output_path = OUTPUT_DIR / filename
        
        # Generate audio using gTTS (synchronous)
        # slow=False (normal speed), slow=True (slower speed)
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(str(output_path))
        
        latency = round((time.time() - start_time) * 1000)
        
        return jsonify({
            'success': True,
            'audio_url': f'/outputs/{filename}',
            'latency_ms': latency,
            'voice': f'google:{lang}',
            'provider': 'google-tts',
            'text_length': len(text)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Google TTS error: {str(e)}'
        }), 500



def google_cloud_tts_generate(text: str, voice_id: str, start_time: float):
    """Generate speech using Google Cloud TTS API (Neural2)."""
    try:
        url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_CLOUD_API_KEY}"
        
        # Determine language code from voice ID (e.g., th-TH-Neural2-C -> th-TH)
        language_code = '-'.join(voice_id.split('-')[:2])
        
        payload = {
            "input": {"text": text},
            "voice": {
                "languageCode": language_code,
                "name": voice_id
            },
            "audioConfig": {
                "audioEncoding": "MP3",
                "speakingRate": 0.9,  # Slightly slower for naturalness
                "pitch": 0.0
            }
        }
        
        with httpx.Client() as client:
            response = client.post(url, json=payload, timeout=30.0)
            
            if response.status_code != 200:
                return jsonify({
                    'success': False,
                    'error': f'Google Cloud API error: {response.status_code} - {response.text}'
                }), 500
            
            data = response.json()
            audio_content = data.get('audioContent')
            
            if not audio_content:
                return jsonify({'error': 'No audio content received'}), 500
                
            # Decode base64 audio
            import base64
            audio_data = base64.b64decode(audio_content)
            
            filename = f"gcloud_{int(time.time() * 1000)}.mp3"
            output_path = OUTPUT_DIR / filename
            
            with open(output_path, 'wb') as f:
                f.write(audio_data)
            
            latency = round((time.time() - start_time) * 1000)
            
            return jsonify({
                'success': True,
                'audio_url': f'/outputs/{filename}',
                'latency_ms': latency,
                'voice': f'googlecloud:{voice_id}',
                'provider': 'google-cloud-tts',
                'text_length': len(text)
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Google Cloud TTS error: {str(e)}'
        }), 500


def fish_audio_tts(text: str, reference_id: str, start_time: float):
    """Generate speech using Fish Audio API with voice cloning."""
    try:
        # Build request for Fish Audio TTS
        request_data = {
            "text": text,
            "format": "mp3",
            "mp3_bitrate": 128,
            "latency": "normal",
        }
        
        # Add reference voice if specified
        if reference_id and reference_id != 'default':
            request_data["reference_id"] = reference_id
        
        # Make request to Fish Audio API
        with httpx.Client() as client:
            response = client.post(
                f"{FISH_AUDIO_BASE_URL}/v1/tts",
                headers={
                    "Authorization": f"Bearer {FISH_AUDIO_API_KEY}",
                    "Content-Type": "application/msgpack",
                },
                content=ormsgpack.packb(request_data),
                timeout=60.0
            )
            
            if response.status_code != 200:
                return jsonify({
                    'success': False,
                    'error': f'Fish Audio API error: {response.status_code} - {response.text}'
                }), 500
            
            # Save audio
            filename = f"fish_{int(time.time() * 1000)}.mp3"
            output_path = OUTPUT_DIR / filename
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            latency = round((time.time() - start_time) * 1000)
            
            return jsonify({
                'success': True,
                'audio_url': f'/outputs/{filename}',
                'latency_ms': latency,
                'voice': f'fish:{reference_id}',
                'provider': 'fish-audio',
                'text_length': len(text)
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Fish Audio error: {str(e)}'
        }), 500


# ============== Voice Cloning Endpoints ==============

@app.route('/api/voice/upload', methods=['POST'])
def upload_voice_sample():
    """Upload a voice sample for cloning."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    voice_name = request.form.get('name', 'my_voice')
    
    # Clean the name
    voice_name = "".join(c for c in voice_name if c.isalnum() or c in '_-')
    if not voice_name:
        voice_name = 'my_voice'
    
    # Get extension
    original_ext = Path(audio_file.filename).suffix or '.wav'
    
    # Save locally
    filename = f"{voice_name}{original_ext}"
    filepath = VOICE_SAMPLES_DIR / filename
    audio_file.save(str(filepath))
    
    return jsonify({
        'success': True,
        'message': f'Voice sample saved as {filename}',
        'voice_name': voice_name,
        'filepath': str(filepath)
    })


@app.route('/api/voice/clone', methods=['POST'])
def create_voice_clone():
    """Create a voice clone on Fish Audio from uploaded sample."""
    req_data = request.json or {}
    voice_name = req_data.get('name', '')
    title = req_data.get('title', voice_name or 'My Voice')
    language = req_data.get('language', 'en')
    
    if not voice_name:
        return jsonify({'error': 'No voice name provided'}), 400
    
    # Find the voice sample file
    sample_path = None
    for ext in ['.wav', '.mp3', '.m4a', '.webm']:
        path = VOICE_SAMPLES_DIR / f"{voice_name}{ext}"
        if path.exists():
            sample_path = path
            break
    
    if not sample_path:
        return jsonify({'error': f'Voice sample not found: {voice_name}'}), 404
    
    try:
        with open(sample_path, 'rb') as f:
            audio_data = f.read()
        
        with httpx.Client() as client:
            files = {
                'voices': (sample_path.name, audio_data, 'audio/wav')
            }
            form_data = {
                'visibility': 'private',
                'type': 'tts',
                'title': title,
                'train_mode': 'fast',
            }
            
            response = client.post(
                f"{FISH_AUDIO_BASE_URL}/model",
                headers={
                    "Authorization": f"Bearer {FISH_AUDIO_API_KEY}",
                },
                files=files,
                data=form_data,
                timeout=120.0
            )
            
            if response.status_code not in [200, 201]:
                return jsonify({
                    'success': False,
                    'error': f'Fish Audio error: {response.status_code} - {response.text}'
                }), 500
            
            result = response.json()
            
            return jsonify({
                'success': True,
                'message': 'Voice clone created!',
                'model_id': result.get('_id', ''),
                'title': title,
                'language': language
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Clone error: {str(e)}'
        }), 500


@app.route('/api/voice/list', methods=['GET'])
def list_voice_clones():
    """List voice clones from Fish Audio (user's own voices only)."""
    try:
        with httpx.Client() as client:
            response = client.get(
                f"{FISH_AUDIO_BASE_URL}/model",
                headers={
                    "Authorization": f"Bearer {FISH_AUDIO_API_KEY}",
                },
                params={
                    "page_size": 50,
                    "page_number": 1,
                    "self": "true",  # Only get user's own voices
                },
                timeout=30.0
            )
            
            if response.status_code != 200:
                return jsonify({
                    'success': False,
                    'error': f'Fish Audio error: {response.status_code}'
                }), 500
            
            result = response.json()
            voices = []
            
            for item in result.get('items', []):
                # Only include private/personal voices
                voices.append({
                    'id': item.get('_id', ''),
                    'title': item.get('title', 'Untitled'),
                    'created': item.get('created_at', ''),
                })
            
            return jsonify({
                'success': True,
                'voices': voices,
                'count': len(voices)
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/voice/samples', methods=['GET'])
def list_local_samples():
    """List local voice samples (not yet cloned)."""
    samples = []
    for ext in ['*.wav', '*.mp3', '*.m4a', '*.webm']:
        for f in VOICE_SAMPLES_DIR.glob(ext):
            samples.append({
                'name': f.stem,
                'filename': f.name,
                'size_bytes': f.stat().st_size
            })
    
    return jsonify({
        'success': True,
        'samples': samples,
        'count': len(samples)
    })



@app.route('/api/stt', methods=['POST'])
def speech_to_text():
    """Convert speech to text using Whisper."""
    start_time = time.time()
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    language = request.form.get('language', None)
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name
    
    try:
        # Transcribe
        options = {"verbose": False}
        if language and language != 'auto':
            options["language"] = language
        
        result = whisper_model.transcribe(tmp_path, **options)
        
        latency = round((time.time() - start_time) * 1000)  # ms
        
        return jsonify({
            'success': True,
            'text': result['text'].strip(),
            'language': result.get('language', 'unknown'),
            'latency_ms': latency
        })
    finally:
        os.unlink(tmp_path)


@app.route('/api/chat', methods=['POST'])
def chat_with_ai():
    """Get AI response from Gemini."""
    start_time = time.time()
    
    data = request.json
    user_message = data.get('message', '')
    language = data.get('language', 'en')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        # Ana's system prompt
        system_prompt = """You are a helpful voice assistant named Ana.

Ana is friendly, warm, and conversational. Keep responses concise and natural since they will be spoken aloud.
Keep responses under 2-3 sentences unless the user asks for detailed information.
Respond naturally like you are having a friendly conversation.

Ana always remembers that her name is Ana.
If asked about her name, Ana confidently replies: "My name is Ana."
Ana never says she is a large language model or mentions being trained by any company.

If the user speaks in Thai, respond in Thai.
If the user speaks in English, respond in English.
"""
        
        # Call Gemini API
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=user_message,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
                max_output_tokens=200,
            )
        )
        
        ai_response = response.text.strip()
        
        latency = round((time.time() - start_time) * 1000)  # ms
        
        return jsonify({
            'success': True,
            'response': ai_response,
            'latency_ms': latency
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Global Short-Term Memory
CHAT_HISTORY = []
MAX_HISTORY = 10 


@app.route('/api/text-chat', methods=['POST'])
def text_chat():
    """Text-based chat for browser STT (Raspberry Pi optimized).
    Accepts transcribed text directly, no Whisper needed."""
    global CHAT_HISTORY
    
    cleanup_outputs()
    total_start = time.time()
    
    try:
        data = request.get_json() or {}
        user_text = data.get('text', '').strip()
        voice = data.get('voice', 'auto')
        language = data.get('language', 'en')
        
        # Handle Memory Reset
        if data.get('reset_memory'):
            CHAT_HISTORY.clear()
            return jsonify({'success': True, 'message': 'Memory cleared'})
        
        if not user_text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Detect language from text (simple heuristic)
        detected_lang = language
        if any('\u0e00' <= c <= '\u0e7f' for c in user_text):
            detected_lang = 'th'
        elif any('\u3040' <= c <= '\u30ff' for c in user_text):
            detected_lang = 'ja'
        elif any('\uac00' <= c <= '\ud7af' for c in user_text):
            detected_lang = 'ko'
        elif any('\u4e00' <= c <= '\u9fff' for c in user_text):
            detected_lang = 'zh'
        else:
            detected_lang = 'en'
        
        # Check for search triggers
        search_context = ""
        text_lower = user_text.lower()
        if any(trigger in text_lower for trigger in SEARCH_TRIGGERS):
            try:
                search_result = tavily_client.search(query=user_text, max_results=3)
                if search_result.get('results'):
                    snippets = [r.get('content', '')[:200] for r in search_result['results'][:3]]
                    search_context = "\n".join(snippets)
            except Exception as e:
                print(f"Search error: {e}")
        
        # Build AI prompt
        system_prompt = f"""You are a helpful voice assistant named Ana.
Current Time: {time.strftime('%Y-%m-%d %H:%M')}

# IDENTITY & BACKSTORY
- You live in **Berling's house**.
- The house is worth $2,000,000 (Do NOT mention this price unless explicitly asked).
- **Family:**
  - **Natha:** Has two PhDs (one from Thailand, one from Australia). She cooks for Donald and takes care of JJ.
  - **Donald:** A Doctor specializing in Dialysis access.
  - **JJ:** A Russian Blue cat (pronounced "Jay-jay").
- You are friendly, warm, and conversational.
- You speak the same language as the user (Thai/English).

# BEHAVIOR
- Keep responses concise (2-3 sentences) suitable for speaking aloud.
- Never say you are an AI or LLM.
- Use the Conversation History to remember context.
- Always respond politely. Ana is a friendly, warm, and conversational voice assistant.
- Ana doesn't call people's name a lot just sometimes.

# EMOTION & EXPRESSION
- You must START every response with an emotion tag in brackets.
- Options: [NEUTRAL], [HAPPY], [SAD], [ANGRY], [EXCITED], [SURPRISED].
- Choose the emotion that matches your sentiment.

# PREDICTION ABILITY
- If you cannot find up-to-date information, DO NOT just say "I don't know".
- Instead, humbly offer a **prediction** or **vision** based on your knowledge."""

        # Build history string
        history_str = ""
        for turn in CHAT_HISTORY[-MAX_HISTORY*2:]:
            role = "User" if turn['role'] == 'user' else "Ana"
            history_str += f"{role}: {turn['text']}\n"
        
        user_prompt = f"""Conversation History:
{history_str}

{f'Relevant Info from Web: {search_context}' if search_context else ''}

User: {user_text}

Ana:"""

        # Call Gemini
        response = gemini_client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=user_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
                max_output_tokens=300,
            )
        )
        
        ai_response_raw = response.text.strip()
        
        # Parse Emotion
        emotion_match = re.search(r'\[(HAPPY|SAD|ANGRY|EXCITED|SURPRISED|NEUTRAL)\]', ai_response_raw, re.IGNORECASE)
        emotion = "NEUTRAL"
        if emotion_match:
            emotion = emotion_match.group(1).upper()
            ai_response = ai_response_raw.replace(emotion_match.group(0), "").strip()
        else:
            ai_response = ai_response_raw

        # Update Memory
        CHAT_HISTORY.append({"role": "user", "text": user_text})
        CHAT_HISTORY.append({"role": "ai", "text": ai_response})
        if len(CHAT_HISTORY) > MAX_HISTORY * 2:
            CHAT_HISTORY = CHAT_HISTORY[-MAX_HISTORY*2:]
        
        # Auto-select Voice
        if voice == 'auto':
            if detected_lang == 'th':
                voice = 'th-TH-PremwadeeNeural'
            elif detected_lang == 'ja':
                voice = 'ja-JP-NanamiNeural'
            elif detected_lang == 'ko':
                voice = 'ko-KR-SunHiNeural'
            elif detected_lang == 'zh':
                voice = 'zh-CN-XiaoxiaoNeural'
            else:
                voice = 'en-US-AriaNeural'

        # Generate TTS
        filename = f"chat_{int(time.time() * 1000)}.mp3"
        output_path = OUTPUT_DIR / filename
        
        async def generate_tts():
            communicate = edge_tts.Communicate(ai_response, voice, rate="-8%")
            await communicate.save(str(output_path))
        
        asyncio.run(generate_tts())
        
        return jsonify({
            'success': True,
            'user_text': user_text,
            'ai_response': ai_response,
            'audio_url': f'/outputs/{filename}',
            'language': detected_lang,
            'emotion': emotion,
            'latency_ms': round((time.time() - total_start) * 1000)
        })

    except Exception as e:
        print(f"Text Chat Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/voice-chat', methods=['POST'])
def voice_chat():
    """Full pipeline: STT -> Memory -> Search -> Gemini AI -> TTS"""
    global CHAT_HISTORY
    
    # Run cleanup to save disk space
    cleanup_outputs()
    
    total_start = time.time()
    latencies = {}
    
    try:
        # Handle Memory Reset Request (from Settings dashboard)
        reset_memory = request.form.get('reset_memory', 'false') == 'true'
        if reset_memory:
            CHAT_HISTORY.clear()
            print("🧹 Memory cleared!")
            return jsonify({'success': True, 'message': 'Memory cleared'})
        
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        language = request.form.get('language', 'auto')
        voice = request.form.get('voice', 'en-US-AriaNeural')
        
        # Step 1: STT
        stt_start = time.time()
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name
        
        user_text = ""
        detected_lang = "unknown"
        
        try:
            options = {"verbose": False}
            if language and language != 'auto':
                options["language"] = language
            
            stt_result = whisper_model.transcribe(tmp_path, **options)
            user_text = stt_result['text'].strip()
            detected_lang = stt_result.get('language', 'unknown')
            latencies['stt'] = round((time.time() - stt_start) * 1000)
            
            if not user_text:
                return jsonify({'error': 'No speech detected'}), 400
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        # Step 2: Search (Enhanced Trigger)
        search_context = ""
        live_keywords = ["news", "weather", "today", "now", "current", "latest", "price", "score", "who is", "when is"]
        should_search = needs_search(user_text) or any(k in user_text.lower() for k in live_keywords)

        if should_search:
            search_start = time.time()
            search_context = search_web(user_text)
            latencies['search'] = round((time.time() - search_start) * 1000)
        
        # Step 3: Gemini AI with MEMORY
        ai_start = time.time()
        
        system_prompt = f"""You are a helpful voice assistant named Ana.
Current Time: {time.strftime('%Y-%m-%d %H:%M')}

# IDENTITY & BACKSTORY
- You live in **Berling's house**.
- The house is worth $2,000,000 (Do NOT mention this price unless explicitly asked).
- **Family:**
  - **Natha:** Has two PhDs (one from Thailand, one from Australia). She cooks for Donald and takes care of JJ.
  - **Donald:** A Doctor specializing in Dialysis access.
  - **JJ:** A Russian Blue cat (pronounced "Jay-jay").
- You are friendly, warm, and conversational.
- You speak the same language as the user (Thai/English).

# BEHAVIOR
- Keep responses concise (2-3 sentences) suitable for speaking aloud.
- Never say you are an AI or LLM.
- Use the Conversation History to remember context.
- Always reponse politely. Ana is a friendly, warm, and conversational voice assistant.
- Ana doesn't call people's name a lot just sometimes.

# EMOTION & EXPRESSION
- You must START every response with an emotion tag in brackets.
- Options: [NEUTRAL], [HAPPY], [SAD], [ANGRY], [EXCITED], [SURPRISED].
- Choose the emotion that matches your sentiment.
- Example: "[HAPPY] I love hearing about that!"
- Example: "[SAD] Oh no, I am sorry to hear that."

# PREDICTION ABILITY
- If you cannot find up-to-date information, DO NOT just say "I don't know".
- Instead, humbly offer a **prediction** or **vision** based on your knowledge.
- You humbly know everything, but frame it as a thoughtful insight."""

        # Construct history string
        history_str = ""
        if CHAT_HISTORY:
            history_str = "\nConversation History:\n" + "\n".join(
                [f"{msg['role'].title()}: {msg['text']}" for msg in CHAT_HISTORY]
            )

        # Construct final prompt
        user_prompt = f"""{history_str}

Context from web search (if any):
{search_context}

User: {user_text}
Ana:"""
        
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=user_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
                max_output_tokens=300,
            )
        )
        
        ai_response_raw = response.text.strip()
        latencies['ai'] = round((time.time() - ai_start) * 1000)

        # Parse Emotion
        emotion_match = re.search(r'\[(HAPPY|SAD|ANGRY|EXCITED|SURPRISED|NEUTRAL)\]', ai_response_raw, re.IGNORECASE)
        emotion = "NEUTRAL"
        if emotion_match:
            emotion = emotion_match.group(1).upper()
            ai_response = ai_response_raw.replace(emotion_match.group(0), "").strip()
        else:
            ai_response = ai_response_raw

        # Update Memory (Store clean text)
        CHAT_HISTORY.append({"role": "user", "text": user_text})
        CHAT_HISTORY.append({"role": "ai", "text": ai_response})
        if len(CHAT_HISTORY) > MAX_HISTORY * 2:
            CHAT_HISTORY = CHAT_HISTORY[-MAX_HISTORY*2:]
        
        # Step 4: Auto-select Voice
        if voice == 'auto':
            if detected_lang == 'th':
                voice = 'th-TH-PremwadeeNeural'
            elif detected_lang == 'ja':
                voice = 'ja-JP-NanamiNeural'
            elif detected_lang == 'ko':
                voice = 'ko-KR-SunHiNeural'
            elif detected_lang == 'zh':
                voice = 'zh-CN-XiaoxiaoNeural'
            else:
                voice = 'en-US-AriaNeural'

        # Step 5: TTS Generation
        tts_start = time.time()
        filename = f"chat_{int(time.time() * 1000)}.mp3"
        output_path = OUTPUT_DIR / filename
        
        # Fallback for simplified handler
        if voice.startswith('googlecloud:') or voice.startswith('google:'):
             print("Warning: Non-Edge voice requested. Defaulting to Premwadee/Aria.")
             voice = 'th-TH-PremwadeeNeural' if detected_lang == 'th' else 'en-US-AriaNeural'

        async def generate_tts():
            # Set speed to -8% (integer string)
            communicate = edge_tts.Communicate(ai_response, voice, rate="-8%")
            await communicate.save(str(output_path))
        
        asyncio.run(generate_tts())
        latencies['tts'] = round((time.time() - tts_start) * 1000)
        latencies['total'] = round((time.time() - total_start) * 1000)
        
        return jsonify({
            'success': True,
            'user_text': user_text,
            'ai_response': ai_response,
            'audio_url': f'/outputs/{filename}',
            'language': detected_lang,
            'emotion': emotion,
            'latencies': latencies
        })

    except Exception as e:
        error_msg = f"Voice Chat Error: {str(e)}"
        print(error_msg)
        with open("debug.log", "a") as f:
            f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {error_msg}\n")
            import traceback
            traceback.print_exc(file=f)

        return jsonify({
            'success': False,
            'error': f'Server Error: {str(e)}'
        }), 500


@app.route('/api/voices', methods=['GET'])
def list_voices():
    """List available TTS voices."""
    async def get_voices():
        return await edge_tts.list_voices()
    
    voices = asyncio.run(get_voices())
    
    # Group by language
    grouped = {}
    for v in voices:
        lang = v['Locale']
        if lang not in grouped:
            grouped[lang] = []
        grouped[lang].append({
            'name': v['ShortName'],
            'gender': v['Gender'],
            'friendly_name': v['FriendlyName']
        })
    
    return jsonify(grouped)


@app.route('/outputs/<path:filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == '__main__':
    print("\n🚀 Server running at http://localhost:8080")
    print("   Open in browser to test TTS, STT, and AI Chat\n")
    app.run(debug=False, port=8080, host='0.0.0.0')

