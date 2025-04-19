import os
import pandas as pd

# Path setup
audio_dir = r"C:\Users\aarth\OneDrive\Desktop\Voice cloning\TTS_Project\dataset\final_audio"
transcript_file = r"C:\Users\aarth\OneDrive\Desktop\Voice cloning\TTS_Project\dataset\merged_transcriptions.txt"
output_csv = r"C:\Users\aarth\OneDrive\Desktop\Voice cloning\TTS_Project\dataset\metadata.csv"

# Load transcripts
with open(transcript_file, "r", encoding="utf-8") as f:
    transcripts = [line.strip() for line in f if line.strip()]

# Get sorted list of .wav files
filenames = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])

# Sanity check
if len(filenames) != len(transcripts):
    raise ValueError(f"Mismatch: {len(filenames)} audio files but {len(transcripts)} transcripts!")

# Create DataFrame
df = pd.read_csv(metadata_file, sep='|', header=None, names=["filename", "text"])


# Save in LJSpeech format: <filename>|<text>
df.to_csv(output_csv, sep='|', index=False, header=False)

print(f"✅ Metadata generated at: {output_csv}")
