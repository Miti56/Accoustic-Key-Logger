import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import noisereduce as nr
import os
from scipy.signal import spectrogram
import buckupData


def load_audio(file_path):
    rate, data = scipy.io.wavfile.read(file_path)
    return rate, data


def save_audio(file_path, rate, data):
    scipy.io.wavfile.write(file_path, rate, np.int16(data / np.max(np.abs(data)) * 32767))


def create_spectrogram(data, rate):
    frequencies, times, Sxx = spectrogram(data, fs=rate, nperseg=1600, noverlap=1536)
    return frequencies, times, Sxx


def plot_spectrogram(times, frequencies, Sxx, title):
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='auto')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(title)
    plt.show()


def reduce_noise(data, sr):
    reduced_noise = nr.reduce_noise(y=data, sr=sr)
    return reduced_noise


def main():
    default_input_directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/app/record/data'
    default_output_directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/app/record/data'
    use_default = input("Do you want to use the default settings? (Y/N): ").upper() == 'Y'
    if use_default:
        input_directory = default_input_directory
        output_directory = default_output_directory
    else:
        input_directory = input("Enter the input directory path: ")
        output_directory = input("Enter the output directory path: ")
    backup_option = input("Do you want to backup the data before processing? (Y/N): ").upper() == 'Y'
    if backup_option:
        buckupData.main()
        print("Data backup complete.")
    os.makedirs(output_directory, exist_ok=True)
    for filename in os.listdir(input_directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(input_directory, filename)
            rate, data = load_audio(file_path)
            frequencies, times, Sxx = create_spectrogram(data, rate)
            plot_spectrogram(times, frequencies, Sxx, "Spectrogram before noise reduction: " + filename)
            reduced_noise = reduce_noise(data, rate)
            output_file_path = os.path.join(output_directory, "reduced_" + filename)
            save_audio(output_file_path, rate, reduced_noise)
            rate2, data2 = load_audio(output_file_path)
            frequencies2, times2, Sxx2 = create_spectrogram(data2, rate2)
            plot_spectrogram(times2, frequencies2, Sxx2, "Spectrogram after noise reduction: " + filename)
    print("Noise reduction complete. The noise reduced signals were saved in the output directory.")


if __name__ == "__main__":
    main()
