import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Directory containing the audio files
directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/clipsCut'

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

        data.append(mfccs_processed)
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
