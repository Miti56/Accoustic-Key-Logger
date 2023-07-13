import os
import numpy as np
import soundfile as sf


def calculate_average_loudness(directory):
    loudness_sum = 0.0
    file_count = 0

    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            audio, _ = sf.read(file_path)
            loudness = np.mean(np.abs(audio))
            loudness_sum += loudness
            file_count += 1

    average_loudness = loudness_sum / file_count
    return average_loudness


def normalize_loudness(directory, target_loudness):
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            audio, sample_rate = sf.read(file_path)
            loudness = np.mean(np.abs(audio))
            normalized_audio = audio * (target_loudness / loudness)
            sf.write(file_path, normalized_audio, sample_rate)


def main():
    directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/app/record/data'
    target_loudness = calculate_average_loudness(directory)
    normalize_loudness(directory, target_loudness)
    print("Loudness normalization completed.")


if __name__ == "__main__":
    main()
