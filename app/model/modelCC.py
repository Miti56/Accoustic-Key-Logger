import os
import librosa
import numpy as np
from scipy.signal import correlate
from sklearn.preprocessing import LabelEncoder
import pickle

def load_audio(file_path):
    audio, sample_rate = librosa.load(file_path)
    return audio, sample_rate

def predict_label(test_data, data, labels):
    # Calculate the cross correlation between the test sample and all samples
    cross_correlations = [correlate(test_data, d) for d in data]
    # Sort the cross correlations by their maximum values
    sorted_indices = np.argsort([np.max(c) for c in cross_correlations])
    # Predict the label
    predicted_label = labels[sorted_indices[-1]]
    return predicted_label

def predict_wav_file(filename, data, labels, le):
    # Load the .wav file
    audio, sample_rate = load_audio(filename)
    # Predict the label
    predicted_label = predict_label(audio, data, labels)
    print(f"The predicted key press for {filename} is {le.inverse_transform([predicted_label])[0]}.")


def accuracy( data, labels):
    # Evaluate the algorithm on the entire data
    correct_predictions = 0
    for i in range(len(data)):
        predicted_label = predict_label(data[i])
        if predicted_label == labels[i]:
            correct_predictions += 1
    # Show accuracy of model
    accuracy = correct_predictions / len(data)
    print(f"Accuracy on entire data: {accuracy}")

def main():
    default_directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/app/record/data'
    use_default = input("Do you want to use the default directory? (Y/N): ").upper() == 'Y'

    if use_default:
        directory = default_directory
    else:
        directory = input("Enter the directory path: ")

    # Load and preprocess the data
    data = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            # Load the .wav file
            audio, sample_rate = load_audio(os.path.join(directory, filename))

            # Extract the label from the filename
            label = filename.split('_')[0]  # adjust this based on how your files are named
            data.append(audio)
            labels.append(label)

    # Convert data and labels to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Encode the labels
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # Save the cross-correlation model and labels
    with open('cross_correlation_model.pkl', 'wb') as f:
        pickle.dump({'dataCC': data, 'labelsCC': labels, 'label_encoderCC': le}, f)

    # Keep predicting WAV files until the user types "quit"
    while True:
        file_path = input("Enter the path of the .wav file to predict (type 'quit' to exit): ")
        if file_path.lower() == 'quit':
            break
        predict_wav_file(file_path, data, labels, le)

if __name__ == "__main__":
    main()
