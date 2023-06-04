import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import noisereduce as nr
import librosa
from scipy.signal import spectrogram

# Load the .wav file
filename = 'recording.wav'
rate, data = scipy.io.wavfile.read(filename)

# Perform a Short-Time Fourier Transform (STFT)
frequencies, times, Sxx = spectrogram(data, fs=rate, nperseg=4096, noverlap=2048)

# Create a plot of the spectrogram
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='auto')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

# Identify the noise - Here we are making an assumption that the first 1000 samples represent noise
noise_part = data[0:1000]
reduced_noise = nr.reduce_noise(y=data, sr=rate)

# Save noise reduced signal to a file
scipy.io.wavfile.write("recordingReduced.wav", rate, np.int16(reduced_noise / np.max(np.abs(reduced_noise)) * 32767))


# Load the .wav file
filename2 = 'recordingReduced.wav'
rate2, data2 = scipy.io.wavfile.read(filename2)
# Perform a Short-Time Fourier Transform (STFT)
frequencies2, times2, Sxx = spectrogram(data2, fs=rate2, nperseg=4096, noverlap=2048)
# Create a plot of the spectrogram
plt.pcolormesh(times2, frequencies2, 10 * np.log10(Sxx), shading='auto')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
print("Noise reduction complete. The noise reduced signal was saved as 'output.wav'.")
