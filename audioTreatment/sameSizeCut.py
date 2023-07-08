import os
import librosa
import numpy as np
import soundfile as sf

# Directory containing the audio files
directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/allClips/clipsForTestingTest3'

# Desired length for the audio signals
desired_length = 1600  # Adjust as needed

# Iterate over the audio files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        # Load the audio file
        file_path = os.path.join(directory, filename)
        audio, sample_rate = librosa.load(file_path)

        # Check the length of the audio signal
        current_length = len(audio)

        # Perform preprocessing if the length is different from the desired length
        if current_length != desired_length:
            if current_length < desired_length:
                # Pad the audio signal with zeros
                pad_length = desired_length - current_length
                audio = np.pad(audio, (0, pad_length), mode='constant')
            else:
                # Truncate the audio signal
                audio = audio[:desired_length]

            # Save the preprocessed audio signal
            sf.write(file_path, audio, sample_rate)

