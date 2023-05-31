import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks


def analyze_recording(input_file):
    # Load the audio recording
    audio, sample_rate = sf.read(input_file)

    # Calculate the duration of the recording
    duration = len(audio) / sample_rate

    # Perform analysis on the audio data
    # Here, we'll use autocorrelation to detect repeating patterns
    autocorr = correlate(audio, audio)
    autocorr /= np.max(autocorr)  # Normalize the autocorrelation values

    # Find the peaks in the autocorrelation signal
    peaks, _ = find_peaks(autocorr, height=0.5)

    # Plot the audio waveform and autocorrelation
    time = np.linspace(0, duration, len(audio))
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, audio)
    plt.title("Audio Recording")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    lag = np.arange(-len(audio) + 1, len(audio))
    plt.plot(lag, autocorr)
    plt.plot(lag[peaks], autocorr[peaks], "ro")
    plt.title("Autocorrelation")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"Recording duration: {duration} seconds")
    print(f"Detected {len(peaks)} repeating patterns")


# Usage example
input_file = 'recording.wav'

analyze_recording(input_file)
