import numpy as np
import tensorflow as tf
import librosa


class MusicLoader(tf.keras.utils.Sequence):

    def __init__(self, batch_size, csv="./fma_small_genres.csv", dir="./data/fma_small/"):
        self.csv = csv
        self.x = np.genfromtxt(csv, delimiter=",", dtype=str)[1:, 0]
        self.y = np.genfromtxt(csv, delimiter=",", dtype=str)[1:, 1]
        self.batch_size = batch_size
        self.dir = dir
        print(str(len(self.x))+" Tracks Loaded")

    def on_epoch_end(self):
        return

    def __getitem__(self, index):
        file_names = self.x[index*self.batch_size: ((index+1)*self.batch_size)]
        X = []

        for file in file_names:
            full_path = self.dir+file[0:3]+"/"+file
            samples, sample_rate = librosa.load(full_path, sr=None)
            sgram = librosa.stft(samples)
            sgram_mag, _ = librosa.magphase(sgram)
            mel_scale_sgram = librosa.feature.melspectrogram(
                S=sgram_mag, sr=sample_rate)
            mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
            X.append((mel_sgram, sample_rate))

        return X, self.y[index*self.batch_size: ((index+1)*self.batch_size)]

    def __len__(self):
        return int(len(self.x)/self.batch_size)
