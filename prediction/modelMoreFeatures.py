import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
import pickle

# Directory containing the audio files
directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/allClips/clipsMechanicalCut'

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

        # Compute other features
        chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        chroma_stft_processed = np.mean(chroma_stft.T, axis=0)

        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        spectral_contrast_processed = np.mean(spectral_contrast.T, axis=0)

        tonnetz = librosa.feature.tonnetz(y=audio, sr=sample_rate)
        tonnetz_processed = np.mean(tonnetz.T, axis=0)

        # Concatenate the features together
        combined_features = np.concatenate(
            [mfccs_processed, chroma_stft_processed, spectral_contrast_processed, tonnetz_processed])

        # Extract the label from the filename
        label = filename.split('_')[0]  # adjust this based on how your files are named

        # data.append(mfccs_processed)
        data.append(combined_features)
        labels.append(label)

# Convert data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Encode the labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# Split the data into training and testing sets
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define the structure of the neural network model
def build_model(input_shape, num_classes):
    """Build a dense neural network model.

    Args:
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of unique labels in the output.

    Returns:
        A Keras Sequential model.
    """
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


# Prepare the model
def compile_and_train(model, train_data, train_labels, test_data, test_labels, epochs=150, batch_size=32):
    """Compile and train the model.

    Args:
        model: The Keras Sequential model.
        train_data: Training data.
        train_labels: Training labels.
        test_data: Testing data.
        test_labels: Testing labels.
        epochs (int, optional): Number of epochs. Defaults to 150.
        batch_size (int, optional): Batch size. Defaults to 32.

    Returns:
        History object containing details about the training process.
    """
    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model and get the training history
    history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size,
                        validation_data=(test_data, test_labels))

    return history


def save_history(history, filename='history.pkl'):
    """Save the model for later evaluation.
            """
    # Save the history object to a file
    with open(filename, 'wb') as f:
        pickle.dump(history.history, f)


def evaluate_model(model, test_data, test_labels):
    """Evaluate the model on the test data.

    Args:
        model: The Keras Sequential model.
        test_data: Testing data.
        test_labels: Testing labels.
    """
    # Evaluate the model
    loss, accuracy = model.evaluate(test_data, test_labels)
    print(f"Test accuracy: {accuracy}")
def predict_key_press(filename, model, le):
    """Prediction of keypresses.

           Args:
               filename: Location of files.
               model: Model built previously.
               le: Predictions.
           """
    # Load the .wav file
    audio, sample_rate = librosa.load(filename)

    # Convert the audio file into MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T,axis=0)

    # Compute other features
    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_stft_processed = np.mean(chroma_stft.T, axis=0)

    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    spectral_contrast_processed = np.mean(spectral_contrast.T, axis=0)

    tonnetz = librosa.feature.tonnetz(y=audio, sr=sample_rate)
    tonnetz_processed = np.mean(tonnetz.T, axis=0)

    # Concatenate the features together
    combined_features = np.concatenate(
        [mfccs_processed, chroma_stft_processed, spectral_contrast_processed, tonnetz_processed])

    # Reshape the data for prediction
    # The model expects input in the form of a batch of samples
    combined_features = combined_features.reshape(1, -1)

    # Use the model to predict the label for the new audio file
    prediction = model.predict(combined_features)

    # Get the index of the highest predicted class
    predicted_index = np.argmax(prediction[0])

    # Convert the predicted index to its corresponding label
    predicted_label = le.inverse_transform([predicted_index])

    # Print out all the classes and their probabilities
    sorted_indices = np.argsort(prediction[0])[::-1]  # get the indices that would sort the array, in descending order
    print("All classes and their probabilities:")
    for idx in sorted_indices:
        print(f"{le.inverse_transform([idx])[0]}: {prediction[0][idx]}")

    return predicted_label[0]



# Use the functions
input_shape = (data_train.shape[1],)  # assuming data_train is already defined
num_classes = len(np.unique(labels))  # assuming labels is already defined
model = build_model(input_shape, num_classes)
history = compile_and_train(model, data_train, labels_train, data_test,
                            labels_test)  # assuming data_test, labels_test are already defined
save_history(history)
evaluate_model(model, data_test, labels_test)

model.save('model.h5')
# Directory containing the audio files
# directory2 = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/clipsCut'
filename = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/allClips/clipsCut/b_6703567.wav'
# for filename in os.listdir(directory2):
predicted_key = predict_key_press(filename, model, le)
print(f"The predicted key press for {filename} is {predicted_key}.")

