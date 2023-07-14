import scipy.io.wavfile
import matplotlib.pyplot as plt
import random
import librosa
import soundfile as sf
from pydub import AudioSegment
import scipy.io.wavfile
from scipy.signal import find_peaks
import os
import numpy as np
import scipy.io.wavfile
from scipy.signal import correlate


def load_audio(filename):
    sample_rate, audio_data = scipy.io.wavfile.read(filename)
    return sample_rate, audio_data


def normalize_audio(audio_data):
    return audio_data / np.max(np.abs(audio_data))


def find_peak_indices(pattern, target, sample_rate_target, peak_threshold=0.245, min_distance=1000):
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
        clip.export(os.path.join(output_directory, f"{peak_time}.wav"), format="wav")

    print("Audio clips created!")



def cut_audio_files(output_directory):
    peak_times_list = []
    combined_loudness = []

    for filename in os.listdir(output_directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(output_directory, filename)

            sample_rate, data = scipy.io.wavfile.read(file_path)

            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            data = data / np.max(np.abs(data))

            peak_index, _ = find_peaks(data, height=0.8)

            peak_time = peak_index / sample_rate
            peak_times_list.append(peak_time)

            combined_loudness.append(np.sqrt(np.mean(data ** 2)))

            if len(peak_index) == 0:
                print(f"No peaks found in {filename}, skipping.")
                os.remove(file_path)
                continue

            first_peak_index = peak_index[0]
            start = int(max(0, first_peak_index - 0.03 * sample_rate))
            end = int(min(len(data), first_peak_index + 0.2 * sample_rate))
            cut_data = data[start:end]

            new_file_path = os.path.join(output_directory, filename)
            scipy.io.wavfile.write(new_file_path, sample_rate, cut_data)



def similarity_cut(output_directory, reference_file):
    ref_sample_rate, ref_data = scipy.io.wavfile.read(reference_file)
    if len(ref_data.shape) > 1:
        ref_data = np.mean(ref_data, axis=1)
    ref_data = ref_data / np.max(np.abs(ref_data))

    # Traverse through all files in the directory
    for filename in os.listdir(output_directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(output_directory, filename)

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
            start = int(max(0, delay - 0.02 * sample_rate))
            end = int(min(len(data), delay + 0.05 * sample_rate))
            cut_data = data[start:end]

            # Copy the file to the new directory and save the cut audio
            new_file_path = os.path.join(output_directory, filename)
            scipy.io.wavfile.write(new_file_path, sample_rate, cut_data)

def size_cut(output_directory, desired_length):
    # Iterate over the audio files in the directory
    print("Some files need some processing...")
    for filename in os.listdir(output_directory):
        if filename.endswith(".wav"):
            # Load the audio file
            file_path = os.path.join(output_directory, filename)
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
    print("Audio processed successfully")

def preprocess_audio(directory, desired_length):
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            audio, sample_rate = librosa.load(file_path)

            current_length = len(audio)

            if current_length != desired_length:
                if current_length < desired_length:
                    pad_length = desired_length - current_length
                    audio = np.pad(audio, (0, pad_length), mode='constant')
                else:
                    audio = audio[:desired_length]

                sf.write(file_path, audio, sample_rate)



def main():
    output_directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/app/test/dataLong'
    # Paths to the pattern and target audio files
    pattern_file = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/app/record/referenceMechanical.wav'
    target_file = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/app/record/fullRecording.wav'
    # 132

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

    # Prompt for plotting option
    option = input("Enter 'P' to only plot or any other key to plot and cut: ").upper()
    if option == 'P':
        # Only perform plotting
        correlation = correlate(target, pattern, mode='valid')
        plot_correlation(correlation, peak_indices)
        return
    else:
        correlation = correlate(target, pattern, mode='valid')
        plot_correlation(correlation, peak_indices)

    # Ask if the user wants to delete the files in the data folder
    delete_files = input("Do you want to delete the files in the data folder (Y/N)? ").upper() == 'Y'

    if delete_files:
        delete_existing_files(output_directory)

    # Create audio clips
    create_audio_clips(target_file, peak_indices, sample_rate_target, output_directory)

    preprocess_audio(output_directory, desired_length=48000)

    # Ask if the user wants to cut using peaks, similarity, or skip cutting step
    cut_method = input("Do you want to cut using peaks (P), similarity (S), or skip cutting (N)? ").upper()

    if cut_method == 'P':
        cut_audio_files(output_directory)
    elif cut_method == 'S':
        use_default_reference = input("Do you want to use the default reference file (Y/N)? ").upper() == 'Y'
        if use_default_reference:
            reference_file = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/app/record/referenceMechanical.wav'
        else:
            reference_file = input("Enter the path to the reference file: ")
        similarity_cut(output_directory, reference_file)
    elif cut_method == 'N':
        print("Skipping cutting step.")
    else:
        print("Invalid choice. Skipping cutting step.")

    # Size Cut
    size_cut(output_directory, desired_length=1600)

    print("Everything finished!")




def delete_existing_files(directory):
    file_list = os.listdir(directory)
    if file_list:
        confirm = input(f"The data folder contains {len(file_list)} files. Are you sure you want to delete them (Y/N)? ").upper() == 'Y'
        if confirm:
            [os.remove(os.path.join(directory, file)) for file in file_list]
            print("Files deleted.")
        else:
            print("Deletion cancelled.")
    else:
        print("No files found in the data folder.")

def get_valid_duration():
    while True:
        duration_input = input("Enter the recording duration in seconds: ")
        try:
            duration = int(duration_input)
            if duration > 0:
                return duration
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")

if __name__ == "__main__":
    main()
