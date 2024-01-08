import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix

import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
import librosa
import numpy as np
import soundfile

AUDIO_FILE = 'FeatureExtraction/StarWars60.wav'

samples, sample_rate = librosa.load(AUDIO_FILE, sr=None)

import librosa.display
import matplotlib.pyplot as plt

# x-axis has been converted to time using our sample rate.
# matplotlib plt.plot(y), would output the same figure, but with sample
# number on the x-axis instead of seconds
plt.figure(figsize=(14, 5))
librosa.display.waveshow(samples, sr=sample_rate)

from IPython.display import Audio
Audio(AUDIO_FILE)

print ('Example shape ', samples.shape, 'Sample rate ', sample_rate, 'Data type', type(samples))
print (samples[22400:22420])

sgram = librosa.stft(samples)
librosa.display.specshow(sgram)

# use the mel-scale instead of raw frequency
sgram_mag, _ = librosa.magphase(sgram)
mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
librosa.display.specshow(mel_scale_sgram)

import numpy as np
# use the decibel scale to get the final Mel Spectrogram
mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
librosa.display.specshow(mel_sgram, sr=sample_rate, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
