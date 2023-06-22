import sounddevice as sd
import numpy as np
from queue import Queue

# Set up a queue for the callback function to put audio data into
q = Queue()

# Set the duration for each audio chunk
duration = 1.0  # in seconds

# Set the sample rate
sample_rate = 48000  # in Hertz

# Calculate the number of frames
frames = int(duration * sample_rate)

# Callback function to capture audio in chunks
def callback(indata, frames, time, status):
    q.put(indata.copy())

# Print available devices and prompt the user to select one
devices = sd.query_devices()
print("Available audio devices:")
for i, device in enumerate(devices):
    print(f"{i}: {device['name']}")
selected_device = int(input("Please enter the number of your preferred audio device: "))

# Threshold for detecting a key press
volume_threshold = 0.05  # adjust as needed

# Set a flag for keypress detection
key_press_detected = False

try:
    with sd.InputStream(device=selected_device, callback=callback, channels=1, blocksize=frames, samplerate=sample_rate):
        print('Recording...')
        while True:
            audio_chunk = q.get().flatten()
            volume = np.max(np.abs(audio_chunk))
            if volume > volume_threshold and not key_press_detected:
                print('Key press detected!')
                key_press_detected = True
            elif volume <= volume_threshold and key_press_detected:
                key_press_detected = False
finally:
    print('Stopped recording')
