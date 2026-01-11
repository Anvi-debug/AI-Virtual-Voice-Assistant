import whisper
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import pyttsx3
from transformers import pipeline
import tempfile

# Load models
asr_model = whisper.load_model("base")
chat_model = pipeline("text-generation", model="microsoft/DialoGPT-small")

# Init speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 180)

def speak(text):
    print(f"🧠 Assistant: {text}")
    engine.say(text)
    engine.runAndWait()

def record_audio(duration=5, fs=16000):
    print("🎙️ Speak now...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    print("✅ Recorded.")
    return np.squeeze(audio)

def transcribe_audio(audio_data, fs=16000):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        write(tmp.name, fs, audio_data)
        result = asr_model.transcribe(tmp.name)
    text = result["text"].strip()
    print(f"👤 You said: {text}")
    return text

def chat_reply(prompt):
    response = chat_model(prompt, max_length=100, pad_token_id=50256)
    text = response[0]["generated_text"].replace(prompt, "").strip()
    return text

def main():
    speak("Hello! I am your offline AI Voice Assistant. How can I help you today?")
    while True:
        audio = record_audio(5)
        text = transcribe_audio(audio)
        if not text:
            continue
        if text.lower() in ["exit", "quit", "bye", "stop"]:
            speak("Goodbye!")
            break
        reply = chat_reply(text)
        speak(reply)

if __name__ == "__main__":
    main()
