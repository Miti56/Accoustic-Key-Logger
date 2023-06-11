import random
import sounddevice as sd
import soundfile as sf
from pynput import keyboard
from pydub import AudioSegment
import time
import os

# List available devices
devices = sd.query_devices()
input_devices = [device['name'] for device in devices if device['max_input_channels'] > 0]

print("Available Microphones:")
for i, device in enumerate(input_devices):
    print(f"{i+1}. {device}")

# Select the device
device_index = int(input("Enter the number of the microphone you want to use: ")) - 1

# Set the recording parameters
fs = 48000  # Sample rate
seconds = 10  # Duration of recording
channels = 1 # Number of channels used

print("Press keys during the recor3ding. Press 'esc' to finish recording.")

keypresses = []
start_time = time.time()

def on_press(key):
    global keypresses
    global start_time
    try:
        print(f'{key} pressed at {time.time()-start_time}')
        keypresses.append((key.char, time.time() - start_time))
    except AttributeError:
        print(f'special key {key} pressed at {time.time()-start_time}')
        keypresses.append((str(key), time.time() - start_time))

listener = keyboard.Listener(on_press=on_press)
listener.start()

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=channels, device=input_devices[device_index])
sd.wait()  # Wait until recording is finished
sf.write('Long1.wav', myrecording, fs)  # Save as WAV file

listener.stop()

print("Creating audio clips...")

song = AudioSegment.from_wav("Long1.wav")

if not os.path.exists("clips"):
    os.makedirs("clips")

for i, (key, press_time) in enumerate(keypresses):
    start_time = int((press_time - 0.5) * 1000)  # convert to ms
    end_time = int((press_time + 0.5) * 1000)  # convert to ms
    clip = song[start_time:end_time]
    random_number = random.randint(1, 10000000)
    clip.export(f"/Users/miti/Documents/GitHub/Accoustic-Key-Logger/record/clips/{key}_{i+random_number}.wav", format="wav")

print("Audio clips created!")
