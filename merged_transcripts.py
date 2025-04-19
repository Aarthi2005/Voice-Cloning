import os

# Define the folder containing transcribed text files
folder_path = "C:\\Users\\aarth\\OneDrive\\Desktop\\Voice cloning\\TTS_Project\\dataset\\final_audio"
output_file = os.path.join(folder_path, "C:\\Users\\aarth\\OneDrive\\Desktop\\Voice cloning\\TTS_Project\\dataset\\merged_transcriptions.txt")

# Find all .txt files in the folder (excluding the merged file itself)
text_files = [f for f in os.listdir(folder_path) if f.endswith(".txt") and f != "merged_transcriptions.txt"]

if not text_files:
    print("⚠️ No text files found to merge!")
else:
    with open(output_file, "w", encoding="utf-8") as outfile:
        for file_name in text_files:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as infile:
                outfile.write(infile.read() + "\n")  # Add newline between transcriptions
            print(f"✅ Merged: {file_name}")

    print(f"🎉 All transcriptions merged into: {output_file}")
