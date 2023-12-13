import numpy as np
import shutil

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
            "155066.mp3",
            "003950.mp3"]

data = np.loadtxt("./fma_small_genres.csv",
                  delimiter=",", dtype=str, skiprows=1)

data_dir = "./data/fma_small/"
new_dir = "./data/fma/"

rock_songs = []
noise_rock_songs = []

for item in data:
    if item[0] in skiplist:
        continue
    if item[1] == 'Rock':
        rock_songs.append(item[0])
    if item[1] == 'Noise-Rock':
        noise_rock_songs.append(item[0])

train_rock_idx, test_rock_idx = round(
    len(rock_songs)*0.8), round(len(rock_songs)*0.2)
train_noise_rock_idx, test_noise_rock_idx = round(
    len(noise_rock_songs)*0.8), round(len(noise_rock_songs)*0.2)

train_rock, test_rock = rock_songs[0:train_rock_idx], rock_songs[train_rock_idx:]
train_noise_rock, test_noise_rock = noise_rock_songs[0:train_noise_rock_idx], noise_rock_songs[train_noise_rock_idx:]

for id in train_rock:
    source = data_dir + id[0:3] + "/" + id
    destination = new_dir+"train_rock/"
    shutil.copy(source, destination)

for id in test_rock:
    source = data_dir + id[0:3] + "/" + id
    destination = new_dir+"test_rock/"
    shutil.copy(source, destination)

for id in train_noise_rock:
    source = data_dir + id[0:3] + "/" + id
    destination = new_dir+"train_noise_rock/"
    shutil.copy(source, destination)

for id in test_noise_rock:
    source = data_dir + id[0:3] + "/" + id
    destination = new_dir+"test_noise_rock/"
    shutil.copy(source, destination)
  