import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import noisereduce as nr
import librosa
import os
from scipy.signal import spectrogram

# Directory containing the audio files
input_directory = 'clips'

# Directory to save the noise reduced audio files
output_directory = 'clipsReduced'
os.makedirs(output_directory, exist_ok=True)

# Traverse through all files in the directory
for filename in os.listdir(input_directory):
    if filename.endswith(".wav"):
        file_path = os.path.join(input_directory, filename)

        # Load the .wav file
        rate, data = scipy.io.wavfile.read(file_path)

        # Perform a Short-Time Fourier Transform (STFT)
        frequencies, times, Sxx = spectrogram(data, fs=rate, nperseg=4096, noverlap=2048)

        # Create a plot of the spectrogram
        plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='auto')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title('Spectrogram before noise reduction: ' + filename)
        plt.show()

        # Identify the noise - Here we are making an assumption that the first 1000 samples represent noise
        noise_part = data[0:1000]
        reduced_noise = nr.reduce_noise(y=data, sr=rate)

        # Save noise reduced signal to a file
        output_file_path = os.path.join(output_directory, "reduced_" + filename)
        scipy.io.wavfile.write(output_file_path, rate, np.int16(reduced_noise / np.max(np.abs(reduced_noise)) * 32767))

        # Load the .wav file
        rate2, data2 = scipy.io.wavfile.read(output_file_path)

        # Perform a Short-Time Fourier Transform (STFT)
        frequencies2, times2, Sxx = spectrogram(data2, fs=rate2, nperseg=4096, noverlap=2048)

        # Create a plot of the spectrogram
        plt.pcolormesh(times2, frequencies2, 10 * np.log10(Sxx), shading='auto')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title('Spectrogram after noise reduction: ' + filename)
        plt.show()

print("Noise reduction complete. The noise reduced signals were saved in the output directory.")
