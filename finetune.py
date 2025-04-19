import os
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
import noisereduce as nr
from TTS.api import TTS
from pydub import AudioSegment
import subprocess
import platform

def convert_to_wav(audio_path):
    if audio_path.lower().endswith(".wav"):
        return audio_path 
    
    wav_path = os.path.splitext(audio_path)[0] + ".wav"
    audio = AudioSegment.from_file(audio_path)
    audio.export(wav_path, format="wav")
    print(f" Converted {audio_path} to WAV: {wav_path}")
    return wav_path

def preprocess_audio(input_wav, output_wav):
    """Preprocess audio: trim silence, reduce noise, normalize."""
    audio, sr = librosa.load(input_wav, sr=22050, mono=True)
    audio_trimmed, _ = librosa.effects.trim(audio)
    audio_denoised = nr.reduce_noise(y=audio_trimmed, sr=sr)
    audio_normalized = audio_denoised / np.max(np.abs(audio_denoised))
    sf.write(output_wav, audio_normalized, sr)
    print(f" Preprocessed and denoised audio saved to: {output_wav}")

'''def load_metadata(metadata_file):
    """Load metadata CSV file."""
    if not os.path.exists(metadata_file):
        print(f" Metadata file '{metadata_file}' not found.")
        return None
    
    df = pd.read_csv(metadata_file)
    print(" Metadata loaded successfully!")
    return df'''

def generate_speech(text, speaker_wav, output_wav):
    """Generate speech with cloned voice."""
    try:
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        tts.tts_to_file(text=text, speaker_wav=speaker_wav, language="en", file_path=output_wav)
        print(f" Generated cloned voice saved to: {output_wav}")
    except TypeError as e:
        print(f" TypeError: {e}. Check parameter usage in tts_to_file.")
    except Exception as e:
        print(f" Unexpected error: {e}")

def fine_tune_xtts_model():

    dataset_path = r"C:\Users\aarth\OneDrive\Desktop\Voice cloning\TTS_Project\dataset"
    output_path = r"C:\Users\aarth\OneDrive\Desktop\Voice cloning\TTS_Project\output"
    pretrained_model_path = r"C:\Users\aarth\OneDrive\Desktop\Voice cloning\TTS_Project\pretrained\xtts_v2"
    config_path = r"C:\Users\aarth\OneDrive\Desktop\Voice cloning\TTS_Project\configs\config_xtts.yaml"

    os.makedirs(output_path, exist_ok=True)

    command = [
        "tts",
        "--continue_path", pretrained_model_path,
        "--config_path", config_path,
        "--data_path", dataset_path,
        "--output_path", output_path
    ]

    print("\n Running XTTS fine-tuning command:")
    print(" ".join(command))

    try:
        if platform.system() == "Windows":
            subprocess.run(command, shell=True)
        else:
            subprocess.run(command)
        print("\n Fine-tuning started. Monitor logs in output directory.")
    except Exception as e:
        print(f" Error during fine-tuning: {e}")

if __name__ == "__main__":
    text = "This is the voice cloning project developed using a pretrained model."
    speaker_audio = r"C:\Users\aarth\Downloads\harvard.wav"
   # metadata_file = r"C:\Users\aarth\OneDrive\Desktop\Voice cloning\TTS_Project\dataset\metadata.csv"'''
    
    speaker_audio = convert_to_wav(speaker_audio)
    preprocessed_wav = "preprocessed_speaker_Harvard.wav"
    output_wav = "voice_Harvard.wav"
    
    preprocess_audio(speaker_audio, preprocessed_wav)
   # metadata = load_metadata(metadata_file)'''
    generate_speech(text, preprocessed_wav, output_wav)