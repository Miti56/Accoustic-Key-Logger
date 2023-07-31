import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
import pickle
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


def load_and_process_data(directory):
    data = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            audio, sample_rate = librosa.load(os.path.join(directory, filename))
            # Convert the audio file into MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, n_fft=1600)
            mfccs_processed = np.mean(mfccs.T, axis=0)

            # # Compute other features
            # chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
            # chroma_stft_processed = np.mean(chroma_stft.T, axis=0)
            #
            # spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate, n_fft=1600)
            # spectral_contrast_processed = np.mean(spectral_contrast.T, axis=0)
            #
            # tonnetz = librosa.feature.tonnetz(y=audio, sr=sample_rate)
            # tonnetz_processed = np.mean(tonnetz.T, axis=0)

            # # Concatenate the features together
            # combined_features = np.concatenate(
            #     [mfccs_processed, chroma_stft_processed, spectral_contrast_processed, tonnetz_processed])
            label = filename.split('_')[0]
            # data.append(combined_features)
            data.append(mfccs_processed)
            labels.append(label)
    data = np.array(data)
    labels = np.array(labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    # Save the label encoder and its classes
    np.save('label_encoder.npy', le.classes_)
    return data, labels, le


def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def compile_and_train(model, train_data, train_labels, test_data, test_labels, epochs=200, batch_size=32):
    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    # Train the model and get the training history
    history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size,
                        validation_data=(test_data, test_labels), callbacks=[early_stopping])

    return history



def compile_and_train_k_fold(model, data, labels, k=5, epochs=200, batch_size=32):
    # Initialize lists to store training and validation metrics for each fold
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Perform k-fold cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []
    for train_index, test_index in kf.split(data):
        train_data, test_data = data[train_index], data[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        # Define early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

        # Train the model and get the training history
        history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size,
                            validation_data=(test_data, test_labels), verbose=0,
                            callbacks=[early_stopping])

        # Accumulate training and validation metrics for each fold
        train_losses.append(history.history['loss'])
        train_accuracies.append(history.history['accuracy'])
        val_losses.append(history.history['val_loss'])
        val_accuracies.append(history.history['val_accuracy'])
        accuracies.append(history.history['val_accuracy'][-1])

    mean_accuracy = np.mean(accuracies)
    print(f"Mean validation accuracy with {k}-fold cross-validation: {mean_accuracy:.4f}")

    # Save the history for each fold
    save_history(history, 'history_k_fold.pkl')

    return model



def save_history(history, filename='history.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(history.history, f)


def evaluate_model(model, test_data, test_labels):
    loss, accuracy = model.evaluate(test_data, test_labels)
    print(f"Test accuracy: {accuracy}")


def predict_key_press(filename, model, le):
    audio, sample_rate = librosa.load(filename)
    # Convert the audio file into MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, n_fft=1600)
    mfccs_processed = np.mean(mfccs.T, axis=0)

    # # Compute other features
    # chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    # chroma_stft_processed = np.mean(chroma_stft.T, axis=0)
    #
    # spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate, n_fft=1600)
    # spectral_contrast_processed = np.mean(spectral_contrast.T, axis=0)
    #
    # tonnetz = librosa.feature.tonnetz(y=audio, sr=sample_rate)
    # tonnetz_processed = np.mean(tonnetz.T, axis=0)
    #
    # combined_features = np.concatenate(
    #     [mfccs_processed, chroma_stft_processed, spectral_contrast_processed, tonnetz_processed], axis=1)
    # # Reshape the data for prediction

    # The model expects input in the form of a batch of samples
    mfccs_processed = mfccs_processed.reshape(1, -1)
    # # Use the model to predict the label for the new audio file
    # prediction = model.predict(combined_features)
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


# def perform_k_fold_training(data, labels, num_classes, folds_list, epochs=200, batch_size=32):
#     fold_accuracies = {}
#     for k in folds_list:
#         print(f"Training with {k}-fold cross-validation...")
#         model = build_model(input_shape, num_classes)
#         model_with_k_fold = compile_and_train_k_fold(model, data, labels, k=k, epochs=epochs, batch_size=batch_size)
#         mean_accuracy = np.mean(model_with_k_fold.history.history['val_accuracy'])
#         print(f"Mean validation accuracy with {k}-fold cross-validation: {mean_accuracy:.4f}")
#         fold_accuracies[k] = mean_accuracy
#
#     return fold_accuracies
#
# def plot_fold_accuracies(fold_accuracies):
#     plt.figure(figsize=(8, 6))
#     plt.plot(list(fold_accuracies.keys()), list(fold_accuracies.values()), marker='o')
#     plt.xlabel('Number of Folds')
#     plt.ylabel('Mean Validation Accuracy')
#     plt.title('Mean Validation Accuracy vs. Number of Folds')
#     plt.grid(True)
#     plt.show()


def main():
    directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/allClips/clipsMechanicalCutResized'
    data, labels, le = load_and_process_data(directory)
    input_shape = (data.shape[1],)
    num_classes = len(np.unique(labels))
    model = build_model(input_shape, num_classes)
    choice = input("Enter 'T' for train-test split or 'K' for k-fold cross-validation: ").upper()

    if choice == 'T':
        data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2,
                                                                            random_state=42)
        history = compile_and_train(model, data_train, labels_train, data_test, labels_test)
        evaluate_model(model, data_test, labels_test)
        model.save('modelTT.h5')
        save_history(history, 'history_train_test_split.pkl')
    elif choice == 'K':
        k = int(input("Enter the number of folds for k-fold cross-validation: "))
        model_with_k_fold = compile_and_train_k_fold(model, data, labels, k=k)
        model_with_k_fold.save('modelKF.h5')
    else:
        print("Invalid choice. Please enter 'T' or 'K'.")

        # directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/allClips/clipsMechanicalCutResized'
        # data, labels, le = load_and_process_data(directory)
        # input_shape = (data.shape[1],)
        # num_classes = len(np.unique(labels))
        # model = build_model(input_shape, num_classes)
        # data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2,
        #                                                                     random_state=42)
        # history = compile_and_train(model, data_train, labels_train, data_test, labels_test)
        # save_history(history)
        # evaluate_model(model, data_test, labels_test)
        # model.save('model.h5')

    # Testing
    default_directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/app/record/unseenData'
    print("To use this option you will need to have previously created unseen data")
    use_default = input("Do you want to use the default directory for testing? (Y/N): ").upper() == 'Y'

    if use_default:
        test_directory = default_directory
    else:
        test_directory = get_directory_input()

    num_files = get_num_files_input()

    # List all the files in test directory
    test_files = os.listdir(test_directory)

    # Iterate over the files
    for filename in test_files[:num_files]:
        if filename.endswith(".wav"):
            # Get the full path of the test file
            file_path = os.path.join(test_directory, filename)

            # Predict the key
            predicted_key = predict_key_press(file_path, model, le)
            print(f"The predicted key press for {filename} is {predicted_key}.")


# Needed outside
directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/allClips/clipsMechanicalCutResized'
data, labels, le = load_and_process_data(directory)
input_shape = (data.shape[1],)
num_classes = len(np.unique(labels))
model = build_model(input_shape, num_classes)
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)
if __name__ == "__main__":
    main()
    # folds_list = [2, 3, 5, 10, 15,20,25,30,35,40,45,60]
    # fold_accuracies = perform_k_fold_training(data, labels, num_classes, folds_list)
    # plot_fold_accuracies(fold_accuracies)
