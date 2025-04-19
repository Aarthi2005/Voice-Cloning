import os
import librosa
import soundfile as sf
from tqdm import tqdm


# Path to your RAVDESS dataset
input_folder = "C:\\Users\\aarth\\OneDrive\\Desktop\\Voice cloning\\TTS_Project\\dataset\\Audio_Speech_Actors_01-24"
output_folder = "C:\\Users\\aarth\\OneDrive\\Desktop\\Voice cloning\\TTS\\dataset_Project\\preprocessed_audio"

os.makedirs(output_folder, exist_ok=True)

def preprocess_audio(file_path, output_path, target_sr=16000):
    """Resamples audio to target sample rate and saves it"""
    y, sr = librosa.load(file_path, sr=None)  # Load original file
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)  # Resample to 16kHz
    sf.write(output_path, y_resampled, target_sr)  # Save processed file

# Process all audio files
for actor_folder in tqdm(os.listdir(input_folder)):
    actor_path = os.path.join(input_folder, actor_folder)
    if os.path.isdir(actor_path):
        for file in os.listdir(actor_path):
            if file.endswith(".wav"):
                input_file_path = os.path.join(actor_path, file)
                output_file_path = os.path.join(output_folder, file)
                preprocess_audio(input_file_path, output_file_path)

print("✅ Audio preprocessing complete! All files are resampled to 16 kHz.")




