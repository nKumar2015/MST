import keras
import librosa
import numpy as np
import tensorflow as tf
from keras.layers import *
import tensorflow_datasets as tfds
from keras.src.utils import losses_utils
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop

dataset = tfds.load('fma', with_info=True, as_supervised=False)[0]
train_style_raw, train_content_raw = dataset['train_style'], dataset['train_content']
test_style_raw, test_content_raw = dataset['test_style'], dataset['test_content']


def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


train_content = [{'genre': item['genre'].numpy(),
                  'sample_rate': item['sample_rate'].numpy(),
                  'sgram': item['sgram'].numpy(),
                  'chromagram': item['chromagram'].numpy(),
                  'path': item['path'].numpy()
                  } for item in train_content_raw]

train_style = [{'genre': item['genre'].numpy(),
                'sample_rate': item['sample_rate'].numpy(),
                'sgram': item['sgram'].numpy(),
                'chromagram': item['chromagram'].numpy(),
                'path': item['path'].numpy()
                } for item in train_style_raw]

test_content = [{'genre': item['genre'].numpy(),
                 'sample_rate': item['sample_rate'].numpy(),
                 'sgram': item['sgram'].numpy(),
                 'chromagram': item['chromagram'].numpy(),
                 'path': item['path'].numpy()
                 } for item in test_content_raw]

test_style = [{'genre': item['genre'].numpy(),
               'sample_rate': item['sample_rate'].numpy(),
               'sgram': item['sgram'].numpy(),
               'chromagram': item['chromagram'].numpy(),
               'path': item['path'].numpy()
               } for item in test_style_raw]



def preprocess(item):
    item['sgram'] = normalize(item['sgram'])
    return item

train_content = [preprocess(item) for item in train_content]
train_style = [preprocess(item) for item in train_style]
test_content = [preprocess(item) for item in test_content]
test_style = [preprocess(item) for item in test_style]


shapedict = {'(128, 2585, 3)': 3349,
             '(128, 2582, 3)': 4216,
             '(128, 2812, 3)': 411, 
             '(128, 1292, 3)': 5, 
             '(128, 1291, 3)': 7
             }

class dataLoader(tf.keras.utils.Sequence):

    def __init__(self, batch_size, train_content, train_style, test_content, test_style):
        self.batch_size = batch_size
        self.train_content = train_content
        self.train_style = train_style
        self.test_content = test_content
        self.test_style = test_style

    def __len__(self):
        return int(len(self.train_content)/self.batch_size)

    def __getitem__(self, index):
        train_style_set = self.train_style[index *
                                           self.batch_size: ((index+1)*self.batch_size)]
        train_content_set = self.train_content[index *
                                               self.batch_size: ((index+1)*self.batch_size)]
        out_x = []
        out_y = []

        for i, val in enumerate(train_content_set):
            content = val
            style = train_style_set[i]

            content_y = (content['path'], content['genre'],
                         content['chromagram'], content['sample_rate'])
            style_y = (style['path'],   style['genre'],
                       style['chromagram'],   style['sample_rate'])

            x = [content['sgram'], style['sgram']]
            y = [content_y, style_y]

            out_x.append(x)
            out_y.append(y)

        return out_x, out_y


class style_content_loss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def convert_to_chromagram(path):
        y, sr = librosa.load(path)
        chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
        return chromagram

    def call(self, y_true, y_pred):
        pred_chromagram = librosa.feature.chroma_stft(S=y_pred)
        true_content_chromagram = y_true[0][3]
        true_style_chromagram = y_true[1][3]

        content_diff = (true_content_chromagram - pred_chromagram) * \
            (true_content_chromagram - pred_chromagram)

        style_dif = (true_style_chromagram - pred_chromagram) * \
            (true_style_chromagram - pred_chromagram)

        return content_diff + style_dif


def Build_model():
    content_input = keras.Input(shape=(128, None, 1))
    style_input   = keras.Input(shape=(128, None, 1))
    
    input = Concatenate()([content_input, style_input])

    x = Conv2D(32, 3, activation="relu")(input)
    x = Conv2D(64, 3, activation="relu")(x)
    block_1_output = MaxPooling2D(3)(x)

    x = Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
    x = Conv2D(64, 3, activation="relu", padding="same")(x)
    block_2_output =  add([x, block_1_output])

    x = Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
    x = Conv2D(64, 3, activation="relu", padding="same")(x)
    block_3_output =  add([x, block_2_output])

    x = Conv2D(64, 3, activation="relu")(block_3_output)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(10)(x)

    model = keras.Model([content_input, style_input], outputs, name="MST")
    return model

model = Build_model()
model.summary()
keras.utils.plot_model(model, "model_arch.png", show_shapes=True)
