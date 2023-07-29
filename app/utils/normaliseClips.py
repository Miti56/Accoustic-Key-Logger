import os
import librosa.display
from pydub import AudioSegment

input_directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/app/record/unseenData'
output_directory = '/Users/miti/Documents/GitHub/Accoustic-Key-Logger/app/record/unseenData'
os.makedirs(output_directory, exist_ok=True)

for filename in os.listdir(input_directory):
    if filename.endswith(".wav"):
        file_path = os.path.join(input_directory, filename)
        audio_data, sample_rate = librosa.load(file_path)
        audio_segment = AudioSegment.from_file(file_path, format="wav")
        normalized_audio_segment = audio_segment.apply_gain(-audio_segment.dBFS)
        normalized_file_path = os.path.join(output_directory, filename)
        normalized_audio_segment.export(normalized_file_path, format="wav")



