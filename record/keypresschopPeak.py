import random
import sounddevice as sd
import soundfile as sf
from pynput import keyboard
from pydub import AudioSegment
import time
import os
import os
import numpy as np
import scipy.io.wavfile
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import shutil

# List available devices
devices = sd.query_devices()
input_devices = [device['name'] for device in devices if device['max_input_channels'] > 0]

print("Available Microphones:")
for i, device in enumerate(input_devices):
    print(f"{i+1}. {device}")

# Select the device
device_index = int(input("Enter the number of the microphone you want to use: ")) - 1

# Set the recording parameters
fs = 48000  # Sample rate
seconds = 120  # Duration of recording
channels = 1 # Number of channels used

print("Press keys during the recor3ding. Press 'esc' to finish recording.")

keypresses = []
start_time = time.time()

def on_press(key):
    global keypresses
    global start_time
    try:
        print(f'{key} pressed at {time.time()-start_time}')
        keypresses.append((key.char, time.time() - start_time))
    except AttributeError:
        print(f'special key {key} pressed at {time.time()-start_time}')
        keypresses.append((str(key), time.time() - start_time))

listener = keyboard.Listener(on_press=on_press)
listener.start()

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=channels, device=input_devices[device_index])
sd.wait()  # Wait until recording is finished
sf.write('Long1.wav', myrecording, fs)  # Save as WAV file

listener.stop()

print("Creating audio clips...")

song = AudioSegment.from_wav("Long1.wav")

if not os.path.exists("clips"):
    os.makedirs("clips")

for i, (key, press_time) in enumerate(keypresses):
    start_time = int((press_time - 0.5) * 1000)  # convert to ms
    end_time = int((press_time + 0.5) * 1000)  # convert to ms
    clip = song[start_time:end_time]
    random_number = random.randint(1, 10000000)
    clip.export(f"/Users/miti/Documents/GitHub/Accoustic-Key-Logger/allClips/clips/{key}_{i+random_number}.wav", format="wav")

print("Audio clips created!")

# Directory containing the audio files
input_directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/allClips/clips'

# Directory to store the processed files
output_directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/allClips/clipsCut'
os.makedirs(output_directory, exist_ok=True)

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

        # If no peaks are found, skip this file
        if len(peak_index) == 0:
            print(f"No peaks found in {filename}, skipping.")
            continue

        # Cut the audio 0.2 seconds before and after the first peak
        first_peak_index = peak_index[0]
        start = int(max(0, first_peak_index - 0.03 * sample_rate))  # make sure we don't go beyond the start of the array
        end = int(
        min(len(data), first_peak_index + 0.1 * sample_rate))  # make sure we don't go beyond the end of the array
        cut_data = data[start:end]

        # Copy the file to the new directory and save the cut audio
        new_file_path = os.path.join(output_directory, filename)
        scipy.io.wavfile.write(new_file_path, sample_rate, cut_data)

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