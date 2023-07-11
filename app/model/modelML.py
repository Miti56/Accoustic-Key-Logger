import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
import pickle


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


def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def compile_and_train(model, train_data, train_labels, test_data, test_labels, epochs=150, batch_size=32):
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size,
                        validation_data=(test_data, test_labels))
    return history


def save_history(history, filename='history.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(history.history, f)


def evaluate_model(model, test_data, test_labels):
    loss, accuracy = model.evaluate(test_data, test_labels)
    print(f"Test accuracy: {accuracy}")


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


def get_directory_input():
    while True:
        directory = input("Enter the directory for testing: ")
        if os.path.isdir(directory):
            return directory
        print("Invalid directory. Please try again.")


def get_num_files_input():
    while True:
        num_files = input("Enter the number of files to run: ")
        if num_files.isdigit():
            return int(num_files)
        print("Invalid input. Please enter a number.")


def main():
    # Directory containing the audio files
    directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/app/record/data'

    data, labels, le = load_and_process_data(directory)

    input_shape = (data.shape[1],)
    num_classes = len(np.unique(labels))

    model = build_model(input_shape, num_classes)

    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    history = compile_and_train(model, data_train, labels_train, data_test, labels_test)

    save_history(history)

    evaluate_model(model, data_test, labels_test)

    model.save('model.h5')

    # Testing
    default_directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/app/record/data'
    use_default = input("Do you want to use the default directory for testing? (Y/N): ").upper() == 'Y'

    if use_default:
        test_directory = default_directory
    else:
        test_directory = get_directory_input()

    num_files = get_num_files_input()

    # List all the files in the test directory
    test_files = os.listdir(test_directory)

    # Iterate over the test files
    for filename in test_files[:num_files]:
        if filename.endswith(".wav"):
            # Get the full path of the test file
            file_path = os.path.join(test_directory, filename)

            # Predict the key press for the test file
            predicted_key = predict_key_press(file_path, model, le)

            # Print the predicted key press for the test file
            print(f"The predicted key press for {filename} is {predicted_key}.")


# Perform train-test split outside the main function
directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/app/record/data'
data, labels, le = load_and_process_data(directory)
input_shape = (data.shape[1],)
num_classes = len(np.unique(labels))
model = build_model(input_shape, num_classes)
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)
if __name__ == "__main__":
    main()
