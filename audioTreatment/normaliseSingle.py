from pydub import AudioSegment

def normalize_audio(input_file, output_file):
    # Load audio file
    audio = AudioSegment.from_file(input_file)

    # Normalize audio to -20 dBFS
    normalized_audio = audio.apply_gain(-audio.dBFS - (-20.0))

    # Save the normalized audio
    normalized_audio.export(output_file, format='wav')

# Usage
normalize_audio('recordingReduced.wav', 'recordingReducedNormalised.wav')
