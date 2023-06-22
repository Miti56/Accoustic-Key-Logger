import os
import numpy as np
import scipy.io.wavfile
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Directory containing the audio files
input_directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/allClips/longWavsMechanicalCut/ACut'

# List to store the peak times of each file
peak_times_list = []

# List to store the combined loudness of all files
combined_loudness = []

# Traverse through all files in the directory
for filename in os.listdir(input_directory):
    if filename.endswith(".wav"):
        file_path = os.path.join(input_directory, filename)

        # Load the .wav file
        sample_rate, data = scipy.io.wavfile.read(file_path)

        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # Normalise the data
        data = data / np.max(np.abs(data))

        # Find the peak
        peak_index, _ = find_peaks(data, height=0.8)  # adjust height based on your needs

        # Convert the peak index to time (seconds)
        peak_time = peak_index / sample_rate
        peak_times_list.append(peak_time)

        # Append the loudness (root mean square) of the data to the combined loudness list
        combined_loudness.append(np.sqrt(np.mean(data**2)))

# Plot the peak times
plt.figure(figsize=(10, 5))
plt.boxplot(peak_times_list, vert=False)
plt.title("Peak times of each .wav file")
plt.xlabel("Time (s)")
plt.show()

# Compute and print the average peak time
average_peak_time = np.mean([item for sublist in peak_times_list for item in sublist])
print(f"Average peak time: {average_peak_time} seconds")

# Plot the combined loudness of all files
plt.figure(figsize=(10, 5))
plt.plot(combined_loudness)
plt.title("Combined loudness of all .wav files")
plt.xlabel("File index")
plt.ylabel("Loudness")
plt.show()
