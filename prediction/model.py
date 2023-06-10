import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

# Directory containing the audio files
directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/clipsCut'

# Load and preprocess the data
data = []
labels = []
for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        # Load the .wav file
        audio, sample_rate = librosa.load(os.path.join(directory, filename))

        # Convert the audio file into MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T,axis=0)

        # Extract the label from the filename
        label = filename.split('_')[0]  # adjust this based on how your files are named

        data.append(mfccs_processed)
        labels.append(label)

# Convert data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Encode the labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# Split the data into training and testing sets
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(data_train.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(np.unique(labels)), activation='softmax'))  # number of unique labels = number of output neurons

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(data_train, labels_train, epochs=50, batch_size=32, validation_data=(data_test, labels_test))

# Evaluate the model
loss, accuracy = model.evaluate(data_test, labels_test)
print(f"Test accuracy: {accuracy}")

def predict_key_press(filename, model, le):
    # Load the .wav file
    audio, sample_rate = librosa.load(filename)

    # Convert the audio file into MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T,axis=0)

    # Reshape the data for prediction
    # The model expects input in the form of a batch of samples
    mfccs_processed = mfccs_processed.reshape(1, -1)

    # Use the model to predict the label for the new audio file
    prediction = model.predict(mfccs_processed)

    # Get the index of the highest predicted class
    predicted_index = np.argmax(prediction[0])

    # Convert the predicted index to its corresponding label
    predicted_label = le.inverse_transform([predicted_index])

    return predicted_label[0]

# Use the function to predict the key press for a new audio file
filename = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/clipsCut/s_35.wav' #s
predicted_key = predict_key_press(filename, model, le)
print(f"The predicted key press for {filename} is {predicted_key}.")

