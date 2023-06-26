import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import sounddevice as sd
from queue import Queue

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

# Load the model
model = load_model('/Users/miti/Documents/GitHub/Accoustic-Key-Logger/prediction/modelSimple.h5')

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

    # # Print out all the classes and their probabilities
    # sorted_indices = np.argsort(prediction[0])[::-1]  # get the indices that would sort the array, in descending order
    # print("All classes and their probabilities:")
    # for idx in sorted_indices:
    #     print(f"{le.inverse_transform([idx])[0]}: {prediction[0][idx]}")


    return predicted_label[0]



# Set up a queue for the callback function to put audio data into
q = Queue()

# Callback function to capture audio in chunks
def callback(indata, frames, time, status):
    q.put(indata.copy())


# Threshold for detecting a key press
volume_threshold = 0.05  # adjust as needed
# Listen and predict function
# def listen_and_predict(model, le, device, duration, noise_duration):
def listen_and_predict(model, le, device, duration):

    # Check for a valid duration
    if duration <= 0:
        print('Invalid duration. Please enter a positive number.')
        return

    # Calculate the number of frames
    frames = int(duration * sample_rate)

    # # Background noise reduction
    # print("Recording background noise profile...")
    # bg_noise = []
    # with sd.InputStream(device=device, channels=1, callback=callback,
    #                     blocksize=frames, samplerate=48000):
    #     for _ in range(int(noise_duration // duration)):
    #         bg_noise.append(q.get().flatten())
    # bg_noise_mean = np.mean(bg_noise)

    try:
        with sd.InputStream(device=device, channels=1, callback=callback,
                            blocksize=frames, samplerate=48000):
            print('Listening...')
            while True:
                # Get the next chunk of audio
                audio_chunk = q.get().flatten()
                # # Subtract the mean of the background noise
                # audio_chunk -= bg_noise_mean
                # # Check if the loudness exceeds the threshold
                volume = np.max(np.abs(audio_chunk))
                if volume > volume_threshold:
                    # Convert the audio file into MFCCs
                    mfccs = librosa.feature.mfcc(y=audio_chunk, sr=sample_rate, n_mfcc=40)
                    mfccs_processed = np.mean(mfccs.T,axis=0)

                    # Reshape the data for prediction
                    mfccs_processed = mfccs_processed.reshape(1, -1)

                    # Use the model to predict the label for the new audio file
                    prediction = model.predict(mfccs_processed)

                    # Get the index of the highest predicted class
                    predicted_index = np.argmax(prediction[0])

                    # Convert the predicted index to its corresponding label
                    predicted_label = le.inverse_transform([predicted_index])

                    print(f"The predicted key press is {predicted_label[0]}.")
                    # # Print out all the classes and their probabilities
                    # sorted_indices = np.argsort(prediction[0])[
                    #                  ::-1]  # get the indices that would sort the array, in descending order
                    # print("All classes and their probabilities:")
                    # for idx in sorted_indices:
                    #     print(f"{le.inverse_transform([idx])[0]}: {prediction[0][idx]}")

    except KeyboardInterrupt:
        print('Stopped listening')


# Print available devices and prompt the user to select one
devices = sd.query_devices()
print("Available audio devices:")
for i, device in enumerate(devices):
    print(f"{i}: {device['name']}")
selected_device = int(input("Please enter the number of your preferred audio device: "))

# Prompt the user for the length of the audio chunks
duration = float(input("Please enter the duration of the audio chunks in seconds: "))

# # Ask the user for how long to record the background noise profile
# noise_duration = float(input("Please enter the duration for recording the background noise profile: "))

# # Start the listening and prediction
# listen_and_predict(model, le, selected_device, duration, noise_duration)

# Start the listening and prediction
listen_and_predict(model, le, selected_device, duration)

