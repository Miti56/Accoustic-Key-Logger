import sounddevice as sd
import librosa
import soundfile as sf


def detect_key_press_sound(input_file, output_file, threshold=0.1):
    # Load the input recording
    audio, sample_rate = librosa.load(input_file, sr=None)

    # Detect key press sound
    key_press_indices = librosa.effects.split(audio, top_db=threshold)

    if len(key_press_indices) == 0:
        print("No key press sound detected.")
        return

    print(f"Detected {len(key_press_indices)} key press sound(s).")

    # Prepare the new recording
    new_recording = []

    # Iterate over the key press sound intervals
    for interval in key_press_indices:
        start_index, end_index = interval

        # Extract the corresponding audio segment
        segment = audio[start_index:end_index]

        # Append the segment to the new recording
        new_recording.extend(segment)

    # Save the new recording to a WAV file
    sf.write(output_file, new_recording, sample_rate)

    print(f"New recording saved to '{output_file}'.")


# Usage example
input_file = 'recording.wav'
output_file = 'filtered_recording.wav'

detect_key_press_sound(input_file, output_file, threshold=0.1)
