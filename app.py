import os
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
from TTS.api import TTS
from pydub import AudioSegment
import streamlit as st
import tempfile

def convert_to_wav(audio_path):
    if audio_path.lower().endswith(".wav"):
        return audio_path
    wav_path = os.path.splitext(audio_path)[0] + ".wav"
    audio = AudioSegment.from_file(audio_path)
    audio.export(wav_path, format="wav")
    return wav_path

def preprocess_audio(input_wav, output_wav):
    audio, sr = librosa.load(input_wav, sr=22050, mono=True)
    audio_trimmed, _ = librosa.effects.trim(audio)
    audio_denoised = nr.reduce_noise(y=audio_trimmed, sr=sr)
    audio_normalized = audio_denoised / np.max(np.abs(audio_denoised))
    sf.write(output_wav, audio_normalized, sr)

def generate_speech(text, speaker_wav, output_path):
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    tts.tts_to_file(text=text, speaker_wav=speaker_wav, language="en", file_path=output_path)

# Streamlit UI
st.set_page_config(page_title="Voice Cloning App", layout="centered")
st.title("🗣️ Voice Cloning with XTTS")
st.markdown("Upload a speaker's audio and enter the text to synthesize a cloned voice.")

# Text input
text_input = st.text_area("Enter text to generate speech:", height=150)

# File upload
audio_file = st.file_uploader("Upload a speaker audio file (WAV, MP3, etc.):", type=["wav", "mp3", "m4a", "ogg"])

# Button to generate speech
if st.button("🎤 Clone Voice and Generate Speech"):
    if audio_file is None or not text_input.strip():
        st.error("Please upload a speaker audio file and enter some text.")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save uploaded audio
            uploaded_audio_path = os.path.join(tmpdir, audio_file.name)
            with open(uploaded_audio_path, "wb") as f:
                f.write(audio_file.read())

            # Convert to WAV if needed
            wav_audio_path = convert_to_wav(uploaded_audio_path)
            preprocessed_path = os.path.join(tmpdir, "preprocessed.wav")
            output_path = os.path.join(tmpdir, "cloned_output.wav")

            # Preprocess and generate
            preprocess_audio(wav_audio_path, preprocessed_path)
            generate_speech(text_input, preprocessed_path, output_path)

            st.success("✅ Voice cloning completed!")

            # 👉 Read the file into memory BEFORE cleanup
            with open(output_path, "rb") as audio_file:
                audio_data = audio_file.read()

# 🔁 Use the audio data after the file and folder are safely cleaned up
        st.audio(audio_data, format="audio/wav")
        st.download_button("📥 Download Cloned Audio", data=audio_data, file_name="cloned_voice.wav")
