import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, GRU, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.models import load_model
from mltu.tensorflow.model_utils import residual_block, activation_layer
from keras import layers
from keras.models import Model
from mltu.configs import BaseModelConfigs
import moviepy.editor as mpy
from mltu.preprocessors import WavReader
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer
import numpy as np



vocab = [x for x in "abcdefghijklmnopqrstuvwxyz "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_model_lip()->Sequential:

    model = Sequential()
    model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(GRU(128, kernel_initializer='Orthogonal', return_sequences=True, name="gru1")))
    model.add(Dropout(.5))

    model.add(Bidirectional(GRU(128, kernel_initializer='Orthogonal', return_sequences=True, name="gru2")))
    model.add(Dropout(.5))

    model.add(Dense(char_to_num.vocabulary_size()+1, kernel_initializer='he_normal', activation='softmax'))

    model.load_weights(os.path.join('..','models','checkpoint'))

    return model


def train_model(input_dim, output_dim, activation="leaky_relu", dropout=0.2):
    
    inputs = layers.Input(shape=input_dim, name="input", dtype=tf.float32)

    # expand dims to add channel dimension
    input = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(inputs)

    # Convolution layer 1
    x = layers.Conv2D(filters=32, kernel_size=[11, 41], strides=[2, 2], padding="same", use_bias=False)(input)
    x = layers.BatchNormalization()(x)
    x = activation_layer(x, activation="leaky_relu")

    # Convolution layer 2
    x = layers.Conv2D(filters=32, kernel_size=[11, 21], strides=[1, 2], padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = activation_layer(x, activation="leaky_relu")
    
    # Reshape the resulted volume to feed the RNNs layers
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)

    # RNN layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    # Dense layer
    x = layers.Dense(256)(x)
    x = activation_layer(x, activation="leaky_relu")
    x = layers.Dropout(dropout)(x)

    # Classification layer
    output = layers.Dense(output_dim + 1, activation="softmax", dtype=tf.float32)(x)
    
    model = Model(inputs=inputs, outputs=output)
    return model
    #audio_model = load_model('../model.h5')



def predict_audio_model(path):
   video = mpy.VideoFileClip(path)
   audio = video.audio
   audio.write_audiofile("audio.wav")
   configs = BaseModelConfigs.load("../models_6_10/content/Models/05_sound_to_text/202310062339/configs.yaml")
   model = train_model(input_dim = configs.input_shape,
   output_dim = len(configs.vocab),
   dropout=0.5)
   model.load_weights("../Models_Dhritiman/Models/05_sound_to_text/202310070653/model.h5")
   spectrogram = WavReader.get_spectrogram("audio.wav", frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length)
   spectrogram = spectrogram[:414,:193]
   padded_spectrogram = np.pad(spectrogram, ((0, configs.max_spectrogram_length - spectrogram.shape[0]),(0,0)), mode="constant", constant_values=0)
   result = model.predict(np.expand_dims(padded_spectrogram, axis=0 ))
   return ctc_decoder(result, configs.vocab)
                                         




