import random
import librosa
import sounddevice as sd
import soundfile as sf
from pynput import keyboard
from pydub import AudioSegment
import time
import scipy.io.wavfile
from scipy.signal import find_peaks
import os
import numpy as np
import scipy.io.wavfile
from scipy.signal import correlate


def select_microphone():
    devices = sd.query_devices()
    input_devices = []
    for device in devices:
        if device['max_input_channels'] > 0:
            input_devices.append(device['name'])

    print("Available microphones:")
    for i, device in enumerate(input_devices):
        print(f"{i + 1}. {device}")

    device_index = int(input("Enter the number of the microphone you want to use:")) - 1
    return input_devices[device_index]


def record_audio(fs, seconds, channels, device):
    print("RECORDING... Press 'ESC' to exit.")
    keypresses = []
    start_time = time.time()

    # Listener for keys and save data
    def on_press(key):
        nonlocal keypresses
        nonlocal start_time
        try:
            print(f'{key} pressed at {time.time() - start_time}')
            keypresses.append((key.char, time.time() - start_time))
        except AttributeError:
            print(f'WARNING!: Unwanted key {key} pressed at {time.time() - start_time}')
            keypresses.append((str(key), time.time() - start_time))

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    longwav = sd.rec(int(seconds * fs), samplerate=fs, channels=channels, device=device)
    sd.wait()
    # Save as WAV file
    sf.write('fullRecording.wav', longwav, fs)
    listener.stop()
    return keypresses


def create_audio_clips(keypresses, output_directory):
    print("Creating audio clips...")

    segment = AudioSegment.from_wav("fullRecording.wav")
    # Create path if not in system
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # Simple cut and export
    for i, (key, press_time) in enumerate(keypresses):
        start_time = int((press_time - 0.5) * 1000)
        end_time = int((press_time + 0.5) * 1000)
        clip = segment[start_time:end_time]
        random_number = random.randint(1, 10000000)
        clip.export(f"{output_directory}/{key}_{i + random_number}.wav", format="wav")

    print("Audio clips created!")


def preprocess_audio(directory, desired_length):
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            audio, sample_rate = librosa.load(file_path)
            current_length = len(audio)
            # Add padding if length is inferior
            if current_length != desired_length:
                if current_length < desired_length:
                    pad_length = desired_length - current_length
                    audio = np.pad(audio, (0, pad_length), mode='constant')
                else:
                    audio = audio[:desired_length]
                # Save
                sf.write(file_path, audio, sample_rate)


def cut_audio_files(output_directory):
    peak_times_list = []
    combined_loudness = []

    for filename in os.listdir(output_directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(output_directory, filename)
            sample_rate, data = scipy.io.wavfile.read(file_path)

            # Mono Audio
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            # Peaks to use for cutting
            data = data / np.max(np.abs(data))
            peak_index, _ = find_peaks(data, height=0.8)
            peak_time = peak_index / sample_rate
            peak_times_list.append(peak_time)
            combined_loudness.append(np.sqrt(np.mean(data ** 2)))

            if len(peak_index) == 0:
                print(f"No peaks found in {filename}, skipping.")
                os.remove(file_path)
                continue

            # Perform the cut
            first_peak_index = peak_index[0]
            start = int(max(0, first_peak_index - 0.03 * sample_rate))
            end = int(min(len(data), first_peak_index + 0.2 * sample_rate))
            cut_data = data[start:end]

            new_file_path = os.path.join(output_directory, filename)
            scipy.io.wavfile.write(new_file_path, sample_rate, cut_data)


def similarity_cut(output_directory, reference_file):
    # Read provided info from sample
    ref_sample_rate, ref_data = scipy.io.wavfile.read(reference_file)
    # Mono audio
    if len(ref_data.shape) > 1:
        ref_data = np.mean(ref_data, axis=1)
    # Normalisation of audio
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

            # Cut the audio 0.2 seconds before and after
            start = int(max(0, delay - 0.02 * sample_rate))
            end = int(min(len(data), delay + 0.05 * sample_rate))
            cut_data = data[start:end]

            # Copy the file to the new directory
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
            # Perform preprocessing if the length is different
            if current_length != desired_length:
                if current_length < desired_length:
                    # Pad the audio signal with zeros
                    pad_length = desired_length - current_length
                    audio = np.pad(audio, (0, pad_length), mode='constant')
                else:
                    # Cut the audio signal
                    audio = audio[:desired_length]
                # Save the preprocessed audio signal
                sf.write(file_path, audio, sample_rate)
    print("Audio processed successfully!")


def main():
    # Ask the user needs to record data
    record_data = input("Do you need to record any data (Y/N)?").upper() == 'Y'

    if not record_data:
        print("Record Section terminated")
        return
    output_directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/app/record/data'

    # Ask if user wants to create a repository of unseen data
    create_unseen_data = input("Do you want to create a repository of unseen data (Y/N)?").upper() == 'Y'

    if create_unseen_data:
        output_directory = input("Enter the output directory for unseen data:")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    # Ask if user wants to delete the files in the data folder
    delete_files = input("Do you want to delete the files in the data folder (Y/N)?").upper() == 'Y'

    if delete_files:
        delete_existing_files(output_directory)

    # Pipeline Record
    device = select_microphone()
    record_duration = input("Enter the recording duration in seconds: ")
    keypresses = record_audio(fs=48000, seconds=record_duration, channels=1, device=device)
    create_audio_clips(keypresses, output_directory)
    preprocess_audio(output_directory, desired_length=48000)

    # Ask if the user wants to cut using peaks, similarity, or skip cutting step
    cut_method = input("Do you want to cut using Peaks (P), Similarity (S), or skip (N)?").upper()

    # Pipeline Cut
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
        print("Skipping cutting step")
    else:
        print("Invalid choice! Skipping cutting step")

    # Size cut verify
    size_cut(output_directory, desired_length=1600)

    print("Everything finished!")


def delete_existing_files(directory):
    file_list = os.listdir(directory)
    if file_list:
        confirm = input(
            f"The data folder contains {len(file_list)} files. Are you sure you want to delete them (Y/N)?").upper() == 'Y'
        if confirm:
            [os.remove(os.path.join(directory, file)) for file in file_list]
            print("Files deleted")
        else:
            print("Deletion cancelled")
    else:
        print("No files found in the data folder")


if __name__ == "__main__":
    main()
