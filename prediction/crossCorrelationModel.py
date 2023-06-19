import os
import librosa
import numpy as np
from scipy.signal import correlate
from sklearn.preprocessing import LabelEncoder
import os
import librosa
import numpy as np
from scipy.signal import correlate
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Directory containing the audio files
directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/allClips/clipsCutSimilarityNorm'

# Load and preprocess the data
data = []
labels = []
for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        # Load the .wav file
        audio, sample_rate = librosa.load(os.path.join(directory, filename))

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

# Function to predict the label of a sample
def predict_label(test_data):
    # Calculate the cross correlation between the test sample and all samples
    cross_correlations = [correlate(test_data, d) for d in data]

    # Get the indices that sort the cross correlations by their maximum values
    sorted_indices = np.argsort([np.max(c) for c in cross_correlations])

    # Predict the label of the test sample to be the same as the label of the sample with the highest cross correlation
    predicted_label = labels[sorted_indices[-1]]

    return predicted_label

# leave-one-out cross-validation approach
# Evaluate the algorithm on the entire data
correct_predictions = 0
for i in range(len(data)):
    predicted_label = predict_label(data[i])
    if predicted_label == labels[i]:
        correct_predictions += 1

accuracy = correct_predictions / len(data)
print(f"Accuracy on entire data: {accuracy}")

# Function to predict the label of a .wav file
def predict_wav_file(filename):
    # Load the .wav file
    audio, sample_rate = librosa.load(filename)

    # Predict the label
    predicted_label = predict_label(audio)

    print(f"The predicted key press for {filename} is {le.inverse_transform([predicted_label])[0]}.")

# Predict a .wav file
filename = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/allClips/clipsCut/a_3620000.wav'
predict_wav_file(filename)

#==============TRAIN-TEST==========================


# Directory containing the audio files
directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/allClips/clipsCutSimilarityNorm'

# Load and preprocess the data
data = []
labels = []
for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        # Load the .wav file
        audio, sample_rate = librosa.load(os.path.join(directory, filename))

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

# Split the data into a train set and a test set
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Function to predict the label of a sample
def predict_label(test_data, data_train):
    # Calculate the cross correlation between the test sample and all train samples
    cross_correlations = [correlate(test_data, d) for d in data_train]

    # Get the indices that sort the cross correlations by their maximum values
    sorted_indices = np.argsort([np.max(c) for c in cross_correlations])

    # Predict the label of the test sample to be the same as the label of the train sample with the highest cross correlation
    predicted_label = labels_train[sorted_indices[-1]]

    return predicted_label

# Evaluate the algorithm on the test set
correct_predictions = 0
for i in range(len(data_test)):
    predicted_label = predict_label(data_test[i], data_train)
    if predicted_label == labels_test[i]:
        correct_predictions += 1

accuracy = correct_predictions / len(data_test)
print(f"Accuracy on test set: {accuracy}")

# Function to predict the label of a .wav file
def predict_wav_file(filename, data_train):
    # Load the .wav file
    audio, sample_rate = librosa.load(filename)

    # Predict the label
    predicted_label = predict_label(audio, data_train)

    print(f"The predicted key press for {filename} is {le.inverse_transform([predicted_label])[0]}.")

# Predict a .wav file
filename = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/allClips/clipsCut/a_3620000.wav'
predict_wav_file(filename, data_train)
