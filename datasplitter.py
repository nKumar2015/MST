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
            "155066.mp3"]

data = np.loadtxt("fma_small_genres.csv", delimiter=",", dtype=str, skiprows=1)

data_dir = "./data/fma_small/"
new_dir = "./data/fma/"

train_portion, test_portion = round(len(data)*0.8), round(len(data)*0.2)

train = data[0:train_portion,:]
test = data[train_portion:,:]

train_content, train_style = np.split(train, 2)
test_content, test_style = np.split(test, 2)

print(train_content)

# Rock to Noise-Rock

for id, genre in train_content:
  if id in skiplist: 
    continue
  
  source = data_dir + id[0:3] + "/" + id
  destination = new_dir+"train_content/"
  shutil.move(source, destination)

for id, genre in train_style:
  if id in skiplist:
    continue

  source = data_dir + id[0:3] + "/" + id
  destination = new_dir+"train_style/"
  shutil.move(source, destination)

for id, genre in test_content:
  if id in skiplist:
    continue

  source = data_dir + id[0:3] + "/" + id
  destination = new_dir+"test_content/"
  shutil.move(source, destination)

for id, genre in test_style:
  if id in skiplist:
    continue

  source = data_dir + id[0:3] + "/" + id
  destination = new_dir+"test_style/"
  shutil.move(source, destination)

