import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model


def load_and_process_data(directory):
    data = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            audio, sample_rate = librosa.load(os.path.join(directory, filename))
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, n_fft=1600)
            mfccs_processed = np.mean(mfccs.T, axis=0)
            label = filename.split('_')[0]
            data.append(mfccs_processed)
            labels.append(label)
    data = np.array(data)
    labels = np.array(labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    return data, labels, le


def predict_key_press(filename, model, le):
    audio, sample_rate = librosa.load(filename)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, n_fft=512)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    mfccs_processed = mfccs_processed.reshape(1, -1)
    prediction = model.predict(mfccs_processed)
    predicted_index = np.argmax(prediction[0])
    predicted_label = le.inverse_transform([predicted_index])
    sorted_indices = np.argsort(prediction[0])[::-1]
    print("All classes and their probabilities:")
    for idx in sorted_indices:
        print(f"{le.inverse_transform([idx])[0]}: {prediction[0][idx]}")
    return predicted_label[0]


def get_file_input():
    while True:
        file_path = input("Enter the path to the WAV file (or 'quit' to exit): ")
        if file_path.lower() == 'quit':
            return None
        if os.path.isfile(file_path):
            return file_path
        print("Invalid file path. Please try again.")


def get_directory_input():
    while True:
        directory_path = input("Enter the path to the directory containing WAV files (or 'quit' to exit): ")
        if directory_path.lower() == 'quit':
            return None
        if os.path.isdir(directory_path):
            return directory_path
        print("Invalid directory path. Please try again.")


def main():
    # Directory containing the audio files
    directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/app/record/data'

    data, labels, le = load_and_process_data(directory)

    # Load the trained model
    model = load_model('/Users/miti/Documents/GitHub/Accoustic-Key-Logger/app/model/model.h5')

    while True:
        print("Choose an option:")
        print("1. Predict a single WAV file")
        print("2. Predict WAV files in a directory")
        print("3. Quit")
        choice = input("Enter your choice: ")

        if choice == '1':
            file_path = get_file_input()
            if file_path is None:
                break
            predicted_key = predict_key_press(file_path, model, le)
            print(f"The predicted key press for {file_path} is {predicted_key}.")
        elif choice == '2':
            directory_path = get_directory_input()
            if directory_path is None:
                break
            for filename in os.listdir(directory_path):
                if filename.endswith(".wav"):
                    file_path = os.path.join(directory_path, filename)
                    predicted_key = predict_key_press(file_path, model, le)
                    print(f"The predicted key press for {file_path} is {predicted_key}.")
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please select a valid option.")


if __name__ == "__main__":
    main()
