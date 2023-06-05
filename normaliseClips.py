import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pydub import AudioSegment

# Directory containing the audio files
input_directory = 'clips'

# Directory to save the normalized audio files
output_directory = 'clipsNormalised'
os.makedirs(output_directory, exist_ok=True)

# Analyze and plot each audio file
for filename in os.listdir(input_directory):
    if filename.endswith(".wav"):
        file_path = os.path.join(input_directory, filename)

        # Load the audio file
        audio_data, sample_rate = librosa.load(file_path)

        # Plot the audio file before normalization
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(audio_data, sr=sample_rate)
        plt.title(f'Waveplot - Before Normalization: {filename}')
        plt.show()

        # Normalize the audio file
        audio_segment = AudioSegment.from_file(file_path, format="wav")
        normalized_audio_segment = audio_segment.apply_gain(-audio_segment.dBFS)

        # Save the normalized audio
        normalized_file_path = os.path.join(output_directory, "normalized_" + filename)
        normalized_audio_segment.export(normalized_file_path, format="wav")

        # Load the normalized audio file
        normalized_audio_data, _ = librosa.load(normalized_file_path)

        # Plot the audio file after normalization
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(normalized_audio_data, sr=sample_rate)
        plt.title(f'Waveplot - After Normalization: normalized_{filename}')
        plt.show()
