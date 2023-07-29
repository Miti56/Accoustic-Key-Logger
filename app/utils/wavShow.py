import librosa
import librosa.display
import matplotlib.pyplot as plt

def plot_waveform_mfcc_chrom_contrast_tonnetz(waveform, sample_rate):
    time = librosa.times_like(waveform, sr=sample_rate)

    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13)

    # Compute Chromagram
    chroma = librosa.feature.chroma_stft(y=waveform, sr=sample_rate)

    # Compute Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=waveform, sr=sample_rate)

    # Compute Tonnetz
    tonnetz = librosa.feature.tonnetz(y=waveform, sr=sample_rate)

    plt.figure(figsize=(18, 12))

    # Plot waveform
    plt.subplot(2, 2, 1)
    plt.plot(time, waveform, color='b')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.grid(True)

    # Find the peak
    peak_idx = waveform.argmax()
    peak_time = time[peak_idx]
    peak_amplitude = waveform[peak_idx]
    plt.annotate(f'Peak: {peak_amplitude:.2f}', xy=(peak_time, peak_amplitude),
                 xytext=(peak_time + 0.1, peak_amplitude + 0.1),
                 arrowprops=dict(arrowstyle='->', color='r'))

    # Plot MFCCs
    plt.subplot(2, 2, 2)
    librosa.display.specshow(mfccs, x_axis='time', sr=sample_rate)
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time (s)')
    plt.ylabel('MFCC')
    plt.title('MFCCs')

    # Plot Chromagram
    plt.subplot(2, 2, 3)
    librosa.display.specshow(chroma, x_axis='time', sr=sample_rate, cmap='coolwarm')
    plt.colorbar()
    plt.xlabel('Time (s)')
    plt.ylabel('Chromagram')
    plt.title('Chromagram')

    plt.subplot(2, 2, 4)
    librosa.display.specshow(contrast, x_axis='time', sr=sample_rate, cmap='coolwarm')
    plt.colorbar()
    plt.xlabel('Time (s)')
    plt.ylabel('Spectral Contrast')
    plt.title('Spectral Contrast')

    plt.suptitle('Waveform, MFCCs, Chromagram, and Spectral Contrast', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main():
    wav_file = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/allClips/clipsMechanicalCutResized/a_2621992.wav'
    waveform, sample_rate = librosa.load(wav_file, sr=48000)
    plot_waveform_mfcc_chrom_contrast_tonnetz(waveform, sample_rate)

if __name__ == "__main__":
    main()
