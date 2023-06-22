import os
import subprocess

def convert_m4a_to_wav(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of all .m4a files in the input folder
    m4a_files = [f for f in os.listdir(input_folder) if f.endswith('.m4a')]

    # Loop through each .m4a file and convert it to .wav
    for m4a_file in m4a_files:
        m4a_path = os.path.join(input_folder, m4a_file)
        wav_file = os.path.splitext(m4a_file)[0] + '.wav'
        wav_path = os.path.join(output_folder, wav_file)

        # Use FFmpeg command to convert .m4a to .wav
        subprocess.run(['ffmpeg', '-i', m4a_path, '-c:a', 'pcm_s16le', wav_path], check=True)

    print("Conversion complete!")

# Specify the input and output folders
input_folder = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/allClips/longWavsMechanical'
output_folder = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/allClips/longWavsMechanicalWav'

# Call the conversion function
convert_m4a_to_wav(input_folder, output_folder)
