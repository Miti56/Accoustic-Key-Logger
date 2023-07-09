import os
import numpy as np
import scipy.io.wavfile
from scipy.signal import correlate
import matplotlib.pyplot as plt
import shutil

# Directory containing the audio files
input_directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/allClips/clipsMechanicalCut'

# Directory to store the processed files
output_directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/allClips/clipsForTestingTest2'
os.makedirs(output_directory, exist_ok=True)

# Reference file for alignment
ref_filename = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/allClips/referenceMechanical.wav'
ref_sample_rate, ref_data = scipy.io.wavfile.read(ref_filename)
if len(ref_data.shape) > 1:
    ref_data = np.mean(ref_data, axis=1)
ref_data = ref_data / np.max(np.abs(ref_data))

# Traverse through all files in the directory
for filename in os.listdir(input_directory):
    if filename.endswith(".wav"):
        file_path = os.path.join(input_directory, filename)

        # Load the .wav file
        sample_rate, data = scipy.io.wavfile.read(file_path)

        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # # Normalise the data
        # data = data / np.max(np.abs(data))

        # Cross-correlation
        corr = correlate(ref_data, data)
        delay = len(data) - np.argmax(corr)

        # Cut the audio 0.2 seconds before and after the peak of correlation
        start = int(max(0, delay - 0.02 * sample_rate))  # make sure we don't go beyond the start of the array
        end = int(min(len(data), delay + 0.05 * sample_rate))  # make sure we don't go beyond the end of the array
        cut_data = data[start:end]

        # Copy the file to the new directory and save the cut audio
        new_file_path = os.path.join(output_directory, filename)
        scipy.io.wavfile.write(new_file_path, sample_rate, cut_data)

