import os
import librosa
import numpy as np

# Paths
input_folder = "C:\\Users\\aarth\\OneDrive\\Desktop\\Voice cloning\\TTS_Project\\dataset\\final_audio"
output_folder = "C:\\Users\\aarth\\OneDrive\\Desktop\\Voice cloning\\TTS_Project\\dataset\\mel_spectrograms"

os.makedirs(output_folder, exist_ok=True)

def convert_to_mel(file_path, output_path, target_sr=16000, n_mels=80, hop_length=256):
    """Converts an audio file to a mel-spectrogram and saves it as .npy."""
    
    # Load and normalize audio
    y, sr = librosa.load(file_path, sr=target_sr)
    y = librosa.util.normalize(y)  # Normalize amplitude
    
    # Compute Mel-Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    
    # Convert to log scale (log-mel spectrogram)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Save mel-spectrogram as numpy array
    np.save(output_path, mel_spec_db)
    print(f"✅ Saved: {output_path}")

# Process all .wav files in input folder
processed_count = 0

for file in os.listdir(input_folder):
    if file.endswith(".wav"):
        input_file_path = os.path.join(input_folder, file)
        output_file_path = os.path.join(output_folder, file.replace(".wav", ".npy"))
        
        try:
            convert_to_mel(input_file_path, output_file_path)
            processed_count += 1
        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

print(f"\n✅ Mel-spectrograms generated and saved for {processed_count} files!")
