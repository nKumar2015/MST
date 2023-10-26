import dataloader
import tensorflow as tf

test = dataloader.MusicLoader(
    10, "./fma_small_genres.csv", "./data/fma_small/")

x = test.getX()

x = tf.constant(x)
