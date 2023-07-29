import sounddevice as sd
import numpy as np
from queue import Queue

q = Queue()
duration = 1.0
sample_rate = 48000
frames = int(duration * sample_rate)

def callback(indata, frames, time, status):
    q.put(indata.copy())
devices = sd.query_devices()
print("Available audio devices:")
for i, device in enumerate(devices):
    print(f"{i}: {device['name']}")
selected_device = int(input("Please enter the number of your preferred audio device:"))
volume_threshold = 0.05
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
    print('Stopped recording!')
