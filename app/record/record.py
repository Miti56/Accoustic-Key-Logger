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
    input_devices = [device['name'] for device in devices if device['max_input_channels'] > 0]

    print("Available Microphones:")
    for i, device in enumerate(input_devices):
        print(f"{i+1}. {device}")

    device_index = int(input("Enter the number of the microphone you want to use: ")) - 1
    return input_devices[device_index]


def record_audio(fs, seconds, channels, device):
    print("Press keys during the recording. Press 'esc' to finish recording.")

    keypresses = []
    start_time = time.time()

    def on_press(key):
        nonlocal keypresses
        nonlocal start_time
        try:
            print(f'{key} pressed at {time.time() - start_time}')
            keypresses.append((key.char, time.time() - start_time))
        except AttributeError:
            print(f'special key {key} pressed at {time.time() - start_time}')
            keypresses.append((str(key), time.time() - start_time))

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=channels, device=device)
    sd.wait()  # Wait until recording is finished
    sf.write('fullRecording.wav', myrecording, fs)  # Save as WAV file

    listener.stop()

    return keypresses


def create_audio_clips(keypresses):
    print("Creating audio clips...")

    song = AudioSegment.from_wav("fullRecording.wav")

    if not os.path.exists("data"):
        os.makedirs("data")

    for i, (key, press_time) in enumerate(keypresses):
        start_time = int((press_time - 0.5) * 1000)  # convert to ms
        end_time = int((press_time + 0.5) * 1000)  # convert to ms
        clip = song[start_time:end_time]
        random_number = random.randint(1, 10000000)
        clip.export(f"data/{key}_{i+random_number}.wav", format="wav")

    print("Audio clips created!")


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
                continue

            first_peak_index = peak_index[0]
            start = int(max(0, first_peak_index - 0.03 * sample_rate))
            end = int(min(len(data), first_peak_index + 0.1 * sample_rate))
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

def main():
    device = select_microphone()
    keypresses = record_audio(fs=48000, seconds=5, channels=1, device=device)
    create_audio_clips(keypresses)
    output_directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/app/record/data'

    preprocess_audio(output_directory, desired_length=48000)

    # Cut using peaks
    cut_audio_files(output_directory)
    # Or cut using similarity
    #similarity_cut(output_directory,reference_file="/Users/miti/Documents/GitHub/Accoustic-Key-Logger/app/record/referenceMechanical.wav")


    # Size Cut
    size_cut(output_directory, desired_length=1600)

    # # Delete files in case of restart
    # folder_path = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/app/record/data'
    # [os.remove(os.path.join(folder_path, file)) for file in os.listdir(folder_path)]
    print("Everything finished!")


if __name__ == "__main__":
    main()