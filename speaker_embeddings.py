import os
import numpy as np
import torchaudio
from speechbrain.inference import SpeakerRecognition

# Set the directory containing audio files
audio_folder = r"C:\Users\aarth\OneDrive\Desktop\Voice cloning\TTS_Project\dataset\final_audio"  

# Load ECAPA-TDNN model for speaker embedding extraction
spk_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    savedir="tmp_ecapa_model"
)

# Dictionary to store embeddings
speaker_embeddings = {}

# Process each audio file in the directory
for filename in os.listdir(audio_folder):
    if filename.endswith(".wav"):  
        file_path = os.path.join(audio_folder, filename)

        try:
            # Load the audio file
            signal, fs = torchaudio.load(file_path)

            # Extract speaker embedding using ECAPA-TDNN
            embedding = spk_model.encode_batch(signal).squeeze().detach().cpu().numpy()
            
            # Store in dictionary
            speaker_embeddings[filename] = embedding.astype(np.float32)  # Ensure correct format

            print(f"✅ Processed: {filename} - Embedding Shape: {embedding.shape}")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

# Path to save the embeddings
embedding_file = r"C:\Users\aarth\OneDrive\Desktop\Voice cloning\TTS_Project\dataset\speaker_embeddings_ecapa.npy"

# Save as a NumPy `.npy` file
np.save(embedding_file, speaker_embeddings)
print(f"✅ Speaker embeddings saved successfully at: {embedding_file}")

# Load the file to verify
loaded_embeddings = np.load(embedding_file, allow_pickle=True).item()
print(f"✅ Loaded Successfully: {type(loaded_embeddings)}")
print(f"🔹 Number of Speakers: {len(loaded_embeddings)}")

# Print shape of one embedding for verification
for filename, embedding in loaded_embeddings.items():
    print(f"🔹 {filename}: {embedding.shape}")
    break  
