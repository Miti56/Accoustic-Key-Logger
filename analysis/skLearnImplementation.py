import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Directory containing the audio files
directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/allClips/clipsMechanicalCut'

# Load and preprocess the data
data = []
filenames = []
for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        # Load the .wav file
        audio, sample_rate = librosa.load(os.path.join(directory, filename))

        # Convert the audio file into MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)

        # # Compute other features
        # chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        # chroma_stft_processed = np.mean(chroma_stft.T, axis=0)
        #
        # spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        # spectral_contrast_processed = np.mean(spectral_contrast.T, axis=0)
        #
        # tonnetz = librosa.feature.tonnetz(y=audio, sr=sample_rate)
        # tonnetz_processed = np.mean(tonnetz.T, axis=0)
        #
        # # Concatenate the features together
        # combined_features = np.concatenate(
        #     [mfccs_processed, chroma_stft_processed, spectral_contrast_processed, tonnetz_processed])

        data.append(mfccs_processed)
        # data.append(combined_features)
        filenames.append(filename)

# Standardize the data to have zero mean and unit variance
scaler = StandardScaler()
data = scaler.fit_transform(np.array(data))

# Perform t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
data_2d = tsne.fit_transform(data)

# Decide on a number of clusters
num_clusters = 26  # adjust based on your knowledge of the data

# Perform K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(data)

# Plot each audio file in the 2D t-SNE space and color it based on its cluster
plt.figure(figsize=(10, 10))
scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=kmeans.labels_)
plt.title('Clusters of audio files')
plt.colorbar(scatter)
plt.show()

# Print out the filenames and their corresponding cluster
for filename, label in zip(filenames, kmeans.labels_):
    print(f"{filename} is in cluster {label}")
