import random
import os
import scipy.io.wavfile
from scipy.signal import correlate, find_peaks
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment


def load_audio(filename):
    sample_rate, audio_data = scipy.io.wavfile.read(filename)
    return sample_rate, audio_data


def normalize_audio(audio_data):
    return audio_data / np.max(np.abs(audio_data))


def find_peak_indices(pattern, target, sample_rate_target, peak_threshold=0.3, min_distance=1000):
    # Cross-correlate the signals
    correlation = correlate(target, pattern, mode='valid')

    # Find peaks in the correlation that are above the peak threshold
    peaks, _ = find_peaks(correlation, height=peak_threshold * np.max(correlation))

    # Group close peaks and compute the average index for each group
    peaks_grouped = []
    current_group = [peaks[0]]

    for i in range(1, len(peaks)):
        if peaks[i] - peaks[i-1] <= min_distance:
            current_group.append(peaks[i])
        else:
            peaks_grouped.append(current_group)
            current_group = [peaks[i]]

    peaks_grouped.append(current_group)

    # Compute average index of close peaks
    average_peaks = [int(np.mean(group)) for group in peaks_grouped]

    # Convert sample indices to times
    peak_times = [peak / sample_rate_target for peak in average_peaks]

    print(f"Found {len(average_peaks)} peaks")
    print(f"Peak times (in seconds): {peak_times}")

    return average_peaks, peak_times


def plot_correlation(correlation, average_peaks):
    plt.plot(correlation)
    plt.plot(average_peaks, correlation[average_peaks], "x")
    plt.title("Cross-correlation")
    plt.show()


def create_audio_clips(audio_file, peak_indices, sample_rate, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    song = AudioSegment.from_wav(audio_file)

    for i, peak_index in enumerate(peak_indices):
        peak_time = peak_index / sample_rate

        start_time = int((peak_time - 0.1) * 1000)  # convert to ms
        end_time = int((peak_time + 0.4) * 1000)  # convert to ms
        clip = song[start_time:end_time]
        random_number = random.randint(1, 10000000)
        clip.export(os.path.join(output_directory, f"a_{peak_time+random_number}.wav"), format="wav")

    print("Audio clips created!")


def main():
    # Paths to the pattern and target audio files
    pattern_file = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/app/record/referenceMechanical.wav'
    target_file = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/app/record/fullRecording.wav'

    # Load the audio files
    sample_rate_pattern, pattern = load_audio(pattern_file)
    sample_rate_target, target = load_audio(target_file)

    # Check if the sample rates match
    assert sample_rate_pattern == sample_rate_target, "Sample rates do not match"

    # Normalize the audio signals
    pattern = normalize_audio(pattern)
    target = normalize_audio(target)

    # Find the peak indices and times
    peak_indices, peak_times = find_peak_indices(pattern, target, sample_rate_target)

    # Plot the cross-correlation and peaks
    correlation = correlate(target, pattern, mode='valid')
    plot_correlation(correlation, peak_indices)


    # # Create audio clips
    # output_directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/allClips/longWavsMechanicalCut'
    # create_audio_clips(target_file, peak_indices, sample_rate_target, output_directory)


if __name__ == "__main__":
    main()
