import os
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from scipy.signal import resample

def process_audio(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    target_sample_rate = 8000
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            input_filepath = os.path.join(input_folder, filename)
            output_filepath = os.path.join(output_folder, filename)
            audio, sample_rate = librosa.load(input_filepath)

            # Packet loss
            packet_loss_rate = 0.2
            audio = drop_packets(audio, packet_loss_rate)

            # Jitter
            jitter_max_shift = 100
            audio = apply_jitter(audio, jitter_max_shift, sample_rate)

            # Codec compression
            resampled_audio = resample(audio, int(len(audio) * target_sample_rate / sample_rate))

            # Bandwidth
            target_sample_rate = 8000
            audio = resample(resampled_audio, int(len(resampled_audio) * target_sample_rate / sample_rate))

            # Save
            sf.write(output_filepath, audio, target_sample_rate)
            print(f"Processed {filename} and saved to {output_filepath}")

def drop_packets(audio, packet_loss_rate):
    return [segment for segment in audio if np.random.rand() > packet_loss_rate]

def apply_jitter(audio, jitter_max_shift, sample_rate):
    shift_amount = int(sample_rate * jitter_max_shift / 1000)
    jittered_audio = np.roll(audio, np.random.randint(-shift_amount, shift_amount))
    return jittered_audio

def compress_audio(audio, sample_rate, target_bitrate):
    audio_segment = AudioSegment(audio.tobytes(), frame_rate=sample_rate, sample_width=audio.dtype.itemsize, channels=1)
    compressed_audio = audio_segment.set_frame_rate(target_bitrate)
    return np.array(compressed_audio.get_array_of_samples())

if __name__ == "__main__":
    input_folder = "/Users/miti/Documents/GitHub/Accoustic-Key-Logger/allClips/clipsCut"
    output_folder = "/Users/miti/Documents/GitHub/Accoustic-Key-Logger/allClips/VoIP"

    process_audio(input_folder, output_folder)
