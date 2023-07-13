import random

import scipy.io.wavfile
from scipy.signal import correlate, find_peaks
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment

import os


# Load the .wav files
sample_rate_pattern, pattern = scipy.io.wavfile.read('/Users/miti/Documents/GitHub/Accoustic-Key-Logger/app/record/referenceMechanical.wav')
sample_rate_target, target = scipy.io.wavfile.read('/Users/miti/Documents/GitHub/Accoustic-Key-Logger/app/record/fullRecording.wav')

# Check if the sample rates match
assert sample_rate_pattern == sample_rate_target, "Sample rates do not match"

# Ensure both signals are in the same range
pattern = pattern / np.max(np.abs(pattern))
target = target / np.max(np.abs(target))

# Cross-correlate the signals
correlation = correlate(target, pattern, mode='valid')

# Find peaks in the correlation that are above a certain height
peak_threshold = 0.3  # adjust this value based on your needs
peaks, _ = find_peaks(correlation, height=peak_threshold * np.max(correlation))

# Define a minimum distance between peaks (in samples)
min_distance = 1000  # adjust this value based on your needs

# To find groups of close peaks and compute the average time for each group:
peaks_grouped = []
current_group = [peaks[0]]

for i in range(1, len(peaks)):
    if peaks[i] - peaks[i-1] <= min_distance:
        current_group.append(peaks[i])
    else:
        peaks_grouped.append(current_group)
        current_group = [peaks[i]]

peaks_grouped.append(current_group)

# Compute average time of close peaks
average_peaks = [int(np.mean(group)) for group in peaks_grouped]

# Convert sample indices to times
peak_times = [peak / sample_rate_target for peak in average_peaks]

# Plot the results
plt.plot(correlation)
plt.plot(average_peaks, correlation[average_peaks], "x")
plt.title("Cross-correlation")
plt.show()

print(f"Found {len(average_peaks)} peaks")
print(f"Peak times (in seconds): {peak_times}")

# print("Creating audio clips...")
# song = AudioSegment.from_wav("/allClips/longWavsMechanicalWav/A.wav")
#
# if not os.path.exists("clipsTest"):
#     os.makedirs("clipsTest")
#
# for i in peak_times:
#     start_time = int((i - 0.1) * 1000)  # convert to ms
#     end_time = int((i + 0.4) * 1000)  # convert to ms
#     clip = song[start_time:end_time]
#     random_number = random.randint(1, 10000000)
#     clip.export(f"/Users/miti/Documents/GitHub/Accoustic-Key-Logger/allClips/longWavsMechanicalCut/{'a'}_{i+random_number}.wav", format="wav")
#
# print("Audio clips created!")
