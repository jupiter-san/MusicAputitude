import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
# Load a flac file from 0(s) to 60(s) and resample to 4.41 KHz
filename = '/home/j/MusicAputitude/music/ARASHI/mp3_data/ARASHI - ｢未完｣ [Official Music Video].mp3'
y, sr = librosa.load(filename, sr=4410, offset=0.0, duration=60.0)
librosa.display.waveshow(y=y, sr=sr)

plt.show()