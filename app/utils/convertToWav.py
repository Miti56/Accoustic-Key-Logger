import os
import subprocess


def convert_audio_to_wav(input_folder, output_folder, input_format):
    os.makedirs(output_folder, exist_ok=True)
    audio_files = [f for f in os.listdir(input_folder) if f.endswith(f".{input_format}")]
    for audio_file in audio_files:
        audio_path = os.path.join(input_folder, audio_file)
        wav_file = os.path.splitext(audio_file)[0] + '.wav'
        wav_path = os.path.join(output_folder, wav_file)

        # Use FFmpeg command to convert audio to WAV
        subprocess.run(['ffmpeg', '-i', audio_path, '-c:a', 'pcm_s16le', wav_path], check=True)

    print("Conversion complete!")


def get_user_input(message, default_value):
    user_input = input(f"{message} (default: {default_value}): ")
    return user_input if user_input != "" else default_value


def main():
    # Default paths and settings
    default_input_folder = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/allClips/longWavsMechanical'
    default_output_folder = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/allClips/longWavsMechanicalWav'
    default_input_format = 'm4a'
    use_default = input("Do you want to use the default settings? (Y/N):").upper() == 'Y'

    if use_default:
        input_folder = default_input_folder
        output_folder = default_output_folder
        input_format = default_input_format
    else:
        input_folder = get_user_input("Enter the input folder path", default_input_folder)
        output_folder = get_user_input("Enter the output folder path", default_output_folder)
        input_format = get_user_input("Enter the input audio format (e.g: mp3, wav)", default_input_format)

    convert_audio_to_wav(input_folder, output_folder, input_format)


if __name__ == "__main__":
    main()
