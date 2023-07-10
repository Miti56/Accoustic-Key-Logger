import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


def load_and_process_data(directory):
    data = []
    filenames = []
    labels = []

    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            audio, sample_rate = librosa.load(os.path.join(directory, filename))
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, n_fft=1600)
            mfccs_processed = np.mean(mfccs.T, axis=0)
            label = filename.split('_')[0]

            data.append(mfccs_processed)
            filenames.append(filename)
            labels.append(label)

    scaler = StandardScaler()
    data = scaler.fit_transform(np.array(data))

    return data, filenames, labels


def perform_tsne(data):
    tsne = TSNE(n_components=2, random_state=42)
    data_2d = tsne.fit_transform(data)
    return data_2d


def perform_clustering(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(data)
    sil_score = silhouette_score(data, kmeans.labels_)
    return kmeans, sil_score


def plot_clusters(data_2d, kmeans, num_clusters):
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=kmeans.labels_)
    plt.title('Clusters of audio files')

    for i in range(num_clusters):
        centroid = np.mean(data_2d[kmeans.labels_ == i], axis=0)
        plt.text(centroid[0], centroid[1], str(i), fontsize=12, color='black')

    plt.colorbar(scatter)
    plt.show()


def print_cluster_information(filenames, kmeans_labels, true_labels):
    # Mapping of cluster labels to letters
    cluster_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
                       10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
                       19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

    for filename, predicted_label, true_label in zip(filenames, kmeans_labels, true_labels):
        predicted_letter = cluster_mapping.get(predicted_label, '?')
        print(f"{filename} is in cluster {predicted_label} ({predicted_letter}), true label: {true_label}")



def analyze_peak_times(directory):
    peak_times_list = []
    combined_loudness = []

    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            sample_rate, data = scipy.io.wavfile.read(file_path)
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            data = data / np.max(np.abs(data))
            peak_index, _ = find_peaks(data, height=0.8)
            peak_time = peak_index / sample_rate
            peak_times_list.append(peak_time)
            combined_loudness.append(np.sqrt(np.mean(data**2)))

    plt.figure(figsize=(10, 5))
    plt.boxplot(peak_times_list, vert=False)
    plt.title("Peak times of each .wav file")
    plt.xlabel("Time (s)")
    plt.show()

    average_peak_time = np.mean([item for sublist in peak_times_list for item in sublist])
    print(f"Average peak time: {average_peak_time} seconds")

    plt.figure(figsize=(10, 5))
    plt.plot(combined_loudness)
    plt.title("Combined loudness of all .wav files")
    plt.xlabel("File index")
    plt.ylabel("Loudness")
    plt.show()


def main():
    directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/app/record/data'

    data, filenames, labels = load_and_process_data(directory)
    data_2d = perform_tsne(data)

    num_clusters = 26
    kmeans, sil_score = perform_clustering(data, num_clusters)
    plot_clusters(data_2d, kmeans, num_clusters)

    print(f"Silhouette score: {sil_score}")
    print_cluster_information(filenames, kmeans.labels_, labels)

    analyze_peak_times(directory)


if __name__ == "__main__":
    main()
