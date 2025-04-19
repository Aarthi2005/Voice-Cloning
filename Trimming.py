import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

# Paths
input_folder = "C:\\Users\\aarth\\OneDrive\\Desktop\\Voice cloning\\TTS_Project\\dataset\\preprocessed_audio"
output_folder = "C:\\Users\\aarth\\OneDrive\\Desktop\\Voice cloning\\TTS_Project\\dataset\\final_audio"

os.makedirs(output_folder, exist_ok=True)

def trim_and_normalize(file_path, output_path, target_db=-20):
    """Trims silence and normalizes audio to a target dB level"""
    
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)  
    
    # Trim silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)  
    
    # Normalize audio (target_db sets the loudness level)
    rms = np.sqrt(np.mean(y_trimmed**2))  # Compute RMS
    scalar = 10**(target_db / 20) / (rms + 1e-8)  # Compute scaling factor
    y_normalized = y_trimmed * scalar  # Apply normalization

    # Save processed audio
    sf.write(output_path, y_normalized, sr)

# Process all preprocessed speech files
processed_count = 0

for file in tqdm(os.listdir(input_folder)):
    if file.endswith(".wav"):
        input_file_path = os.path.join(input_folder, file)
        output_file_path = os.path.join(output_folder, file)
        trim_and_normalize(input_file_path, output_file_path)
        processed_count += 1

print(f"✅ Silence trimmed and audio normalized for {processed_count} files!")
