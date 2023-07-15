import subprocess
import random
import sys

import librosa
import sounddevice as sd
import soundfile as sf
from pynput import keyboard
from pydub import AudioSegment
import time
import scipy.io.wavfile
from scipy.signal import find_peaks
import os
import numpy as np
import scipy.io.wavfile
from scipy.signal import correlate

def display_prompt(prompt):
    print(prompt, end=' ')
    return input()

def run_record():
    try:
        subprocess.run(['python3', 'record/record.py'], check=True)
    except subprocess.CalledProcessError:
        print("Error running the original program.")

def run_data():
    try:
        subprocess.run(['python3', 'DataVisualisation/dataVisualisation.py'], check=True)
    except subprocess.CalledProcessError:
        print("Error running the original program.")

def main():
    print("Welcome to the UI!")

    q1 = display_prompt("Are you in a quiet space to start the recording? ").upper() == 'Y'

    if q1:
        # Run the original program
        run_data()
    else:
        q2 = display_prompt("Do you wish to continue with already existent data? ").upper() == 'Y'
        if q2:
            pass
        else:
            print("Program terminated.")
            sys.exit()
        print("Redirecting...")
        time.sleep(3)


    q3 = display_prompt("Visualise Data? ").upper() == 'Y'

    if q3:
        # Run the original program
        run_data()
    else:
        print("Program terminated.")


if __name__ == "__main__":
    main()

