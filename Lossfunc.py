import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

dataset = tfds.load('fma', with_info=True, as_supervised=False)[0]
train_style_raw, train_content_raw = dataset['train_style'], dataset['train_content']
test_style_raw, test_content_raw = dataset['test_style'], dataset['test_content']

def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image

train_content = [{'genre': item['genre'].numpy,
                  'genre': item['sample_rate'].numpy,
                  'sgram_image': item['sgram_image'].numpy,
                  } for item in train_content_raw]

train_style   = [{'genre': item['genre'].numpy,
                  'genre': item['sample_rate'].numpy,
                  'sgram_image': item['sgram_image'].numpy,
                  } for item in train_style_raw]

test_content  = [{'genre': item['genre'].numpy,
                  'genre': item['sample_rate'].numpy,
                  'sgram_image': item['sgram_image'].numpy,
                  } for item in test_content_raw]

test_style    = [{'genre': item['genre'].numpy,
                  'genre': item['sample_rate'].numpy,
                  'sgram_image': item['sgram_image'].numpy,
                  } for item in test_style_raw]

def preprocess(item):
    item['sgram_image'] = normalize('sgram_image')
    return item

train_content = [preprocess(item) for item in train_content]
train_style   = [preprocess(item) for item in train_style]
test_content  = [preprocess(item) for item in test_content]
test_style    = [preprocess(item) for item in test_style]

