import os
import random
import pygame
from pynput import keyboard
from functools import partial


def play_random_sound(sound_folder, sound_files):
    sound_file = random.choice(sound_files)
    sound_path = os.path.join(sound_folder, sound_file)
    sound = pygame.mixer.Sound(sound_path)
    sound.play()


def on_press(sound_folder, sound_files, key):
    # Play a random sound when a key is pressed
    play_random_sound(sound_folder, sound_files)


def on_release(key):
    # Stop the listener
    if key == keyboard.Key.esc:
        return False


def play_keyboard_sounds(sound_folder, sound_files):
    # Initialize pygame mixer
    pygame.mixer.init()
    on_press_partial = partial(on_press, sound_folder, sound_files)
    # Create a keyboard listener
    listener = keyboard.Listener(on_press=on_press_partial, on_release=on_release)
    listener.start()
    # Keep the main thread running while the listener is active
    listener.join()


def main():
    # Folder path containing the keyboard sound files
    sound_folder = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/app/utils/fakeKeyboardSounds'

    # Ask if the user has replaced the sounds in the folder
    replace_sounds = input("Have you replaced the sounds in the folder? (Y/N): ").upper()

    if replace_sounds == "Y":
        # Load all sound files from the folder
        sound_files = os.listdir(sound_folder)

        # Call the function to play keyboard sounds
        play_keyboard_sounds(sound_folder, sound_files)
    else:
        # Use a default folder with random WAV samples
        default_folder = 'app/record/data'
        sound_files = os.listdir(default_folder)

        if len(sound_files) < 5:
            print("Insufficient sound files in the default folder.")
        else:
            # Randomly select 5 sound files from the default folder
            random_sound_files = random.sample(sound_files, 5)
            sound_folder = default_folder

            # Call the function to play keyboard sounds
            play_keyboard_sounds(sound_folder, random_sound_files)


if __name__ == "__main__":
    main()
