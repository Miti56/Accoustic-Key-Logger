import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# Directory containing the audio files
directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/allClips/clipsMechanicalCutResized'

# Load and preprocess the data
data = []
filenames = []
labels = []
for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        # Load the .wav file
        audio, sample_rate = librosa.load(os.path.join(directory, filename))

        # Convert the audio file into MFCCs
        #mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, n_fft=1600)
        mfccs_processed = np.mean(mfccs.T, axis=0)

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

        # Extract the true label from the filename
        label = filename.split('_')[0]  # adjust this based on how your files are named

        data.append(combined_features)
        # data.append(mfccs_processed)
        filenames.append(filename)
        labels.append(label)

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

# Calculate the silhouette score
sil_score = silhouette_score(data, kmeans.labels_)
print(f"Silhouette score: {sil_score}")

# Plot each audio file in the 2D t-SNE space and color it based on its cluster
plt.figure(figsize=(10, 10))
scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=kmeans.labels_)
plt.title('Clusters of audio files')

# Add cluster labels to plot
for i in range(num_clusters):
    centroid = np.mean(data_2d[kmeans.labels_ == i], axis=0)
    plt.text(centroid[0], centroid[1], str(i), fontsize=12, color='black')

plt.colorbar(scatter)
plt.show()

# Print out the filenames, their corresponding cluster, and their true label
for filename, predicted_label, true_label in zip(filenames, kmeans.labels_, labels):
    print(f"{filename} is in cluster {predicted_label} (true label: {true_label})")
