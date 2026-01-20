// ===== TTS Application =====

class TTSApp {
    constructor() {
        // Check browser support
        if (!('speechSynthesis' in window)) {
            alert('Sorry, your browser does not support Text-to-Speech. Please try Chrome, Edge, or Safari.');
            return;
        }

        this.synth = window.speechSynthesis;
        this.voices = [];
        this.utterance = null;
        this.isPaused = false;

        // DOM Elements
        this.textInput = document.getElementById('text-input');
        this.voiceSelect = document.getElementById('voice-select');
        this.rateSlider = document.getElementById('rate-slider');
        this.rateValue = document.getElementById('rate-value');
        this.pitchSlider = document.getElementById('pitch-slider');
        this.pitchValue = document.getElementById('pitch-value');
        this.volumeSlider = document.getElementById('volume-slider');
        this.volumeValue = document.getElementById('volume-value');
        this.speakBtn = document.getElementById('speak-btn');
        this.pauseBtn = document.getElementById('pause-btn');
        this.stopBtn = document.getElementById('stop-btn');
        this.statusBar = document.getElementById('status-bar');
        this.statusText = this.statusBar.querySelector('.status-text');
        this.visualizer = document.getElementById('visualizer');
        this.charCounter = document.getElementById('char-counter');

        this.init();
    }

    init() {
        // Load voices
        this.loadVoices();
        
        // Chrome requires this event listener for voices
        if (speechSynthesis.onvoiceschanged !== undefined) {
            speechSynthesis.onvoiceschanged = () => this.loadVoices();
        }

        // Event listeners
        this.textInput.addEventListener('input', () => this.updateCharCount());
        this.rateSlider.addEventListener('input', () => this.updateSliderDisplay());
        this.pitchSlider.addEventListener('input', () => this.updateSliderDisplay());
        this.volumeSlider.addEventListener('input', () => this.updateSliderDisplay());
        
        this.speakBtn.addEventListener('click', () => this.speak());
        this.pauseBtn.addEventListener('click', () => this.togglePause());
        this.stopBtn.addEventListener('click', () => this.stop());

        // Initial updates
        this.updateCharCount();
        this.updateSliderDisplay();

        // Handle page visibility to prevent speech issues
        document.addEventListener('visibilitychange', () => {
            if (document.hidden && this.synth.speaking) {
                this.synth.pause();
            }
        });
    }

    loadVoices() {
        this.voices = this.synth.getVoices();
        
        // Clear existing options
        this.voiceSelect.innerHTML = '';

        if (this.voices.length === 0) {
            this.voiceSelect.innerHTML = '<option>No voices available</option>';
            return;
        }

        // Group voices by language
        const voiceGroups = {};
        this.voices.forEach((voice, index) => {
            const lang = voice.lang.split('-')[0].toUpperCase();
            if (!voiceGroups[lang]) {
                voiceGroups[lang] = [];
            }
            voiceGroups[lang].push({ voice, index });
        });

        // Create optgroups
        Object.keys(voiceGroups).sort().forEach(lang => {
            const optgroup = document.createElement('optgroup');
            optgroup.label = this.getLanguageName(lang);
            
            voiceGroups[lang].forEach(({ voice, index }) => {
                const option = document.createElement('option');
                option.value = index;
                option.textContent = `${voice.name} (${voice.lang})`;
                if (voice.default) {
                    option.selected = true;
                }
                optgroup.appendChild(option);
            });
            
            this.voiceSelect.appendChild(optgroup);
        });

        // Select first English voice if available
        const englishVoice = this.voices.findIndex(v => v.lang.startsWith('en'));
        if (englishVoice !== -1) {
            this.voiceSelect.value = englishVoice;
        }

        this.updateStatus('ready', `${this.voices.length} voices loaded`);
    }

    getLanguageName(code) {
        const languages = {
            'EN': '🇬🇧 English',
            'ES': '🇪🇸 Spanish',
            'FR': '🇫🇷 French',
            'DE': '🇩🇪 German',
            'IT': '🇮🇹 Italian',
            'PT': '🇵🇹 Portuguese',
            'RU': '🇷🇺 Russian',
            'ZH': '🇨🇳 Chinese',
            'JA': '🇯🇵 Japanese',
            'KO': '🇰🇷 Korean',
            'AR': '🇸🇦 Arabic',
            'HI': '🇮🇳 Hindi',
            'TH': '🇹🇭 Thai',
            'VI': '🇻🇳 Vietnamese',
            'NL': '🇳🇱 Dutch',
            'PL': '🇵🇱 Polish',
            'TR': '🇹🇷 Turkish',
            'SV': '🇸🇪 Swedish',
            'DA': '🇩🇰 Danish',
            'NO': '🇳🇴 Norwegian',
            'FI': '🇫🇮 Finnish',
            'EL': '🇬🇷 Greek',
            'HE': '🇮🇱 Hebrew',
            'ID': '🇮🇩 Indonesian',
            'MS': '🇲🇾 Malay',
            'CS': '🇨🇿 Czech',
            'SK': '🇸🇰 Slovak',
            'UK': '🇺🇦 Ukrainian',
            'RO': '🇷🇴 Romanian',
            'HU': '🇭🇺 Hungarian',
            'BG': '🇧🇬 Bulgarian',
            'HR': '🇭🇷 Croatian',
            'CA': '🏴 Catalan',
        };
        return languages[code] || `🌐 ${code}`;
    }

    updateCharCount() {
        const count = this.textInput.value.length;
        this.charCounter.textContent = count.toLocaleString();
    }

    updateSliderDisplay() {
        this.rateValue.textContent = this.rateSlider.value;
        this.pitchValue.textContent = this.pitchSlider.value;
        this.volumeValue.textContent = Math.round(this.volumeSlider.value * 100);
    }

    speak() {
        const text = this.textInput.value.trim();
        
        if (!text) {
            this.updateStatus('error', 'Please enter some text to speak');
            this.shakeElement(this.textInput);
            return;
        }

        // Stop any current speech
        this.synth.cancel();

        // Create new utterance
        this.utterance = new SpeechSynthesisUtterance(text);
        
        // Set voice
        const selectedVoice = this.voices[this.voiceSelect.value];
        if (selectedVoice) {
            this.utterance.voice = selectedVoice;
        }

        // Set parameters
        this.utterance.rate = parseFloat(this.rateSlider.value);
        this.utterance.pitch = parseFloat(this.pitchSlider.value);
        this.utterance.volume = parseFloat(this.volumeSlider.value);

        // Event handlers
        this.utterance.onstart = () => {
            this.updateStatus('speaking', 'Speaking...');
            this.setButtonStates(true);
            this.showVisualizer(true);
            this.isPaused = false;
            this.updatePauseButton();
        };

        this.utterance.onend = () => {
            this.updateStatus('ready', 'Finished speaking');
            this.setButtonStates(false);
            this.showVisualizer(false);
            this.isPaused = false;
            this.updatePauseButton();
        };

        this.utterance.onerror = (event) => {
            if (event.error !== 'canceled') {
                this.updateStatus('error', `Error: ${event.error}`);
                console.error('Speech error:', event);
            }
            this.setButtonStates(false);
            this.showVisualizer(false);
        };

        this.utterance.onpause = () => {
            this.updateStatus('paused', 'Paused');
        };

        this.utterance.onresume = () => {
            this.updateStatus('speaking', 'Speaking...');
        };

        // Chrome bug workaround: speech stops after ~15 seconds
        // This keeps it going by periodically checking
        this.chromeBugWorkaround();

        // Speak!
        this.synth.speak(this.utterance);
    }

    chromeBugWorkaround() {
        // Chrome has a bug where TTS stops after ~15 seconds
        // This workaround periodically pauses/resumes to keep it going
        const resumeInterval = setInterval(() => {
            if (!this.synth.speaking) {
                clearInterval(resumeInterval);
                return;
            }
            this.synth.pause();
            this.synth.resume();
        }, 10000);
    }

    togglePause() {
        if (!this.synth.speaking) return;

        if (this.isPaused) {
            this.synth.resume();
            this.isPaused = false;
            this.showVisualizer(true);
        } else {
            this.synth.pause();
            this.isPaused = true;
            this.showVisualizer(false);
        }
        this.updatePauseButton();
    }

    stop() {
        this.synth.cancel();
        this.updateStatus('ready', 'Stopped');
        this.setButtonStates(false);
        this.showVisualizer(false);
        this.isPaused = false;
        this.updatePauseButton();
    }

    updateStatus(state, message) {
        this.statusBar.className = 'status-bar ' + state;
        this.statusText.textContent = message;
    }

    setButtonStates(speaking) {
        this.pauseBtn.disabled = !speaking;
        this.stopBtn.disabled = !speaking;
    }

    updatePauseButton() {
        const icon = this.pauseBtn.querySelector('.btn-icon');
        const text = this.pauseBtn.querySelector('.btn-text');
        
        if (this.isPaused) {
            icon.textContent = '▶';
            text.textContent = 'Resume';
        } else {
            icon.textContent = '⏸';
            text.textContent = 'Pause';
        }
    }

    showVisualizer(show) {
        if (show) {
            this.visualizer.classList.remove('hidden');
        } else {
            this.visualizer.classList.add('hidden');
        }
    }

    shakeElement(element) {
        element.style.animation = 'shake 0.5s ease-in-out';
        setTimeout(() => {
            element.style.animation = '';
        }, 500);
    }
}

// Add shake animation dynamically
const style = document.createElement('style');
style.textContent = `
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        20%, 60% { transform: translateX(-5px); }
        40%, 80% { transform: translateX(5px); }
    }
`;
document.head.appendChild(style);

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new TTSApp();
});
