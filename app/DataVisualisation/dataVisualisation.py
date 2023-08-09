import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.signal import find_peaks
import scipy.io.wavfile
import os


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
    n_init = 10
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=n_init).fit(data)
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


def extract_mfccs(filename):
    audio, sample_rate = librosa.load(filename)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, n_fft=1600)
    return mfccs


def preprocess_mfccs(mfccs):
    scaler = StandardScaler()
    mfccs_processed = np.mean(mfccs.T, axis=0)
    mfccs_processed = scaler.fit_transform(mfccs_processed.reshape(1, -1))
    return mfccs_processed


def print_cluster_information(filenames, kmeans_labels, true_labels):
    num_clusters = np.max(kmeans_labels) + 1
    cluster_mapping = {}
    for i in range(num_clusters):
        cluster_mapping[i] = guess_cluster_label(kmeans_labels, i, true_labels)
    print_complete_information = input("Do you want to see the complete cluster information (Y/N)?").upper() == 'Y'

    if print_complete_information:
        for filename, predicted_label, true_label in zip(filenames, kmeans_labels, true_labels):
            predicted_letter = cluster_mapping.get(predicted_label, '?')
            print(f"{filename} is in cluster {predicted_label} ({predicted_letter}), true label: {true_label}")
    else:
        print("Cluster information not printed")


def guess_cluster_label(kmeans_labels, cluster_label, true_labels):
    cluster_data = [true_label for true_label, predicted_label in zip(true_labels, kmeans_labels) if
                    predicted_label == cluster_label]
    if cluster_data:
        most_common_label = max(set(cluster_data), key=cluster_data.count)
        return most_common_label
    else:
        return '?'


def evaluate_average_peak_time(average_peak_time):
    print(f"Average peak time: {average_peak_time} seconds")
    if average_peak_time <= 0.1:
        print("The peak times are well-synchronized")
    elif average_peak_time <= 0.3:
        print("The peak times are reasonably synchronized")
    elif average_peak_time <= 0.5:
        print("The peak times show some synchronization, but there may be variations")
    elif average_peak_time <= 1.0:
        print("The peak times have weak synchronization and significant variations")
    else:
        print("The peak times are poorly synchronized or there may be timing issues")


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
            combined_loudness.append(np.sqrt(np.mean(data ** 2)))

    plt.figure(figsize=(10, 5))
    plt.boxplot(peak_times_list, vert=False)
    plt.title("Peak times of each .wav file")
    plt.xlabel("Time (ms)")
    plt.show()
    average_peak_time = np.mean([item for sublist in peak_times_list for item in sublist])
    evaluate_average_peak_time(average_peak_time)
    plt.figure(figsize=(10, 5))
    plt.plot(combined_loudness)
    plt.title("Combined loudness of all .wav files")
    plt.xlabel("File index")
    plt.ylabel("Loudness")
    plt.show()
    evaluate_combined_loudness(combined_loudness)


def evaluate_combined_loudness(combined_loudness):
    print("Combined Loudness:")

    if len(set(combined_loudness)) == 1:
        print("The combined loudness is homogeneous across all files")
    else:
        print("The combined loudness varies across files")


def evaluate_silhouette_score(silhouette_score):
    print(f"Silhouette score: {silhouette_score}")

    if silhouette_score >= 0.7:
        print("The clusters are well-separated and distinct")
    elif silhouette_score >= 0.5:
        print("The clusters are reasonably well-separated")
    elif silhouette_score >= 0.3:
        print("The clusters show some separation, but there may be overlap")
    elif silhouette_score >= 0.1:
        print("The clusters have weak separation and significant overlap")
    else:
        print("The clusters are poorly separated or there may be incorrect assignments")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    default_directory = os.path.join(parent_dir, 'record', 'data')
    while True:
        use_default = input("Do you want to use the default directory (Y/N)?").upper() == 'Y'
        if use_default:
            directory = default_directory
        else:
            directory = input("Enter the directory path:")
        try:
            data, filenames, labels = load_and_process_data(directory)
            data_2d = perform_tsne(data)
            num_clusters = 26
            kmeans, sil_score = perform_clustering(data, num_clusters)
            plot_clusters(data_2d, kmeans, num_clusters)
            evaluate_silhouette_score(sil_score)
            print_cluster_information(filenames, kmeans.labels_, labels)
            analyze_peak_times(directory)
            break
        except FileNotFoundError:
            print("Directory not found. Please make sure the directory exists and try again")


if __name__ == "__main__":
    main()
