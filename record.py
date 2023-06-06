import sounddevice as sd
import soundfile as sf


def show_available_microphones():
    # Get a list of available microphones
    microphones = sd.query_devices()
    print("Available microphones:")
    for i, microphone in enumerate(microphones):
        print(f"{i+1}. {microphone['name']}")

    # Ask the user to select a microphone
    selection = input("Select a microphone (enter the corresponding number): ")
    microphone_index = int(selection) - 1
    selected_microphone = microphones[microphone_index]

    return selected_microphone


def start_recording(sample_rate=48000, duration=20, channels=1):
    selected_microphone = show_available_microphones()
    microphone_index = selected_microphone['index']
    microphone_name = selected_microphone['name']

    print(f"Recording from microphone '{microphone_name}'...")

    # Start recording audio
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, blocking=True,
                       device=microphone_index)

    # Save the recording to a WAV file
    file_path = 'Q.wav'
    sf.write(file_path, recording, sample_rate)

    print(f"Recording saved to {file_path}")


# Usage example
start_recording(sample_rate=48000, duration=20, channels=1)
