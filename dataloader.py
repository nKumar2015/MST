import numpy as np
import tensorflow as tf
import librosa

skiplist = ["011298.mp3",
            "021657.mp3",
            "029245.mp3",
            "054568.mp3",
            "054576.mp3",
            "098565.mp3",
            "098567.mp3",
            "098569.mp3",
            "099234.mp3",
            "099134.mp3",
            "108925.mp3",
            "133297.mp3",
            "155066.mp3"]

class MusicLoader(tf.keras.utils.Sequence):

    def __init__(self, batch_size, csv="./fma_small_genres.csv", data_dir="./data/fma_small/"):
        self.csv = csv
        self.data_dir = data_dir
        self.x = self.loadsgrams(np.genfromtxt(
            csv, delimiter=",", dtype=str)[1:, 0])
        self.y = np.genfromtxt(csv, delimiter=",", dtype=str)[1:, 1]
        self.batch_size = batch_size
        print(str(len(self.x))+" Tracks Loaded")

    def on_epoch_end(self):
        return

    def loadsgrams(self, files):
        sgrams = []
        for file in files:
            if file in skiplist:
                continue 
            print("loading: "+file)
            full_path = self.data_dir+file[0:3]+"/"+file
            samples, sample_rate = librosa.load(full_path, sr=None)
            sgram = librosa.stft(samples)
            sgram_mag, _ = librosa.magphase(sgram)
            mel_scale_sgram = librosa.feature.melspectrogram(
                S=sgram_mag, sr=sample_rate)
            mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
            sgrams.append((mel_sgram, sample_rate))
        return sgrams

    def __getitem__(self, index):
        return self.x[index*self.batch_size: ((index+1)*self.batch_size)], self.y[index*self.batch_size: ((index+1)*self.batch_size)]

    def __len__(self):
        return int(len(self.x)/self.batch_size)
