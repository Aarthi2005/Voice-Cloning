# Voice-Cloning

This project demonstrates **voice cloning**, which is the process of synthesizing speech that sounds like a specific person. It uses deep learning models to clone a voice from a short audio sample and generate new speech that mimics the original speaker’s tone, pitch, and speaking style.

# Problem Statement

To develop a deep learning-based voice cloning system that can mimic a speaker’s voice using a short audio sample and generate realistic, personalized speech. The system uses the XTTS-v2 model and is fine-tuned on speaker data from the RAVDESS dataset to produce expressive, high-quality speech that retains the speaker’s identity and emotional tone.

# Abstract

Voice cloning has emerged as a powerful application of deep learning in speech synthesis. This project implements voice cloning using the XTTS-v2 model, a multilingual, zero-shot TTS system. We fine-tuned the pretrained model using speaker data from the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset, which contains high-quality speech recordings expressed with various emotions.

The pipeline includes audio preprocessing, metadata creation, speaker embedding extraction, and model fine-tuning. Once trained, the system is capable of generating natural-sounding speech in the target speaker’s voice for any input text.

# Dataset – RAVDESS
The project uses the RAVDESS dataset, which contains:

 24 professional actors (12 male, 12 female)

7356 files in total (speech and song)

Emotion-rich speech samples including neutral, calm, happy, sad, angry, fearful, disgust, and surprised

 High-quality audio in WAV format (48 kHz)

For our voice cloning task:

We selected only speech samples (not songs)

Used recordings from a single speaker for fine-tuning

Trimmed and downsampled to 16 kHz for compatibility with XTTS-v2

# 🛠️ Methods & Implementation
1. Preprocessing:
   
Trimmed silence and normalized audio with preprocessing.py and trimming.py

Downsampled to 22050 Hz, mono-channel

Transcripts auto-generated or extracted, saved to metadata.csv

2. Speaker Embedding:

Used speaker_embedding.py to extract speaker identity features

The embedding represents voice characteristics like pitch, tone, and speaking style

3. Fine-tuning XTTS-v2:

Ran finetune.py using the prepared dataset

Used pretrained XTTS-v2 model as base

Optimized the decoder and vocoder on speaker-specific data

4. Inference / Speech Synthesis:
   
After training, generated cloned voice using:

tts.tts_to_file(text="Hello, how are you?", speaker_wav="sample.wav", file_path="output.wav")

# Results
Successfully generated cloned speech that mimics the RAVDESS speaker’s voice

Speech retains emotion, pronunciation, and pacing from the training speaker

Cloned speech was intelligible and sounded natural



# 📢 Output
![Screenshot 2025-04-17 181717](https://github.com/user-attachments/assets/0cb2f073-6168-4853-ba19-5b7afd263121)


# 🎙️ Reference voice
Listen to the [sample.wav](https://github.com/Aarthi2005/Voice-Cloning/blob/main/voice_Harvard.wav)

# 🗣️ Cloned voice
Listen to the [cloned.wav](https://github.com/Aarthi2005/Voice-Cloning/blob/main/cloned_voice.wav)



