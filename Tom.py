import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import IPython
from utility import *
import subprocess

from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
from sklearn.metrics import f1_score

# for taking voice input from user
from voice_input import take_input

def create_model(input_shape):   
    X_input = Input(shape = input_shape)
    
    # Step 1: CONV layer : for extracting features
    X = Conv1D(filters = 196, kernel_size = 15, strides=4)(X_input)         # CONV1D
    X = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(X)        # Batch normalization
    X = Activation('relu')(X)                                               # ReLu activation
    X = Dropout(0.8)(X)                                                     # Dropout using a 0.8 criteria

    # Step 2: First GRU Layer
    X = GRU(units = 128, return_sequences = True)(X)                     # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                                  # Dropout using a 0.8 criteria
    X = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(X)     # Batch normalization
    
    # Step 3: Second GRU Layer
    X = GRU(units = 128, return_sequences = True)(X)                 # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                              # Dropout using a 0.8 criteria
    X = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(X)  # Batch normalization
    X = Dropout(0.8)(X)                                               # Dropout using a 0.8 criteria
    
    # Step 4: Time-distributed dense layer 
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X)        # Time distributed (sigmoid)

    model = Model(inputs = X_input, outputs = X)
    
    return model  

def detect_triggerword(filename, model):
    plt.subplot(2, 1, 1)
    # Spectogram Generation of input voice
    x = graph_spectrogram(filename)  
    x  = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    return predictions


def action_on_activate(chime_file, filename, predictions, threshold):
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]
    consecutive_timesteps = 0
    for i in range(Ty):
        consecutive_timesteps += 1
        if predictions[0,i,0] > threshold and consecutive_timesteps > 75:
            audio_clip = audio_clip.overlay(chime, position = ((i / Ty) * audio_clip.duration_seconds)*1000)
            consecutive_timesteps = 0
            print('Detected Trigger Word\n')
            subprocess.call(r'./Jerry.command')
        
    audio_clip.export("./resources/detected_output.wav", format='wav')


# Preprocess the audio to the correct format
def preprocess_audio(filename):
    padding = AudioSegment.silent(duration=10000)
    segment = AudioSegment.from_wav(filename)[:10000]
    segment = padding.overlay(segment)
    segment = segment.set_frame_rate(44100)
    segment.export(filename, format='wav')


def main():
    Tx = 5511
    Ty = 1375
    n_freq = 101

    model = create_model(input_shape=(Tx, n_freq))
    model = load_model('./resources/models/tr_model.h5')

    chime_file = "./resources/audio_examples/chime.wav"

    os.system('cls' if os.name == 'nt' else 'clear')

    file_name = take_input()
    preprocess_audio(file_name)
    alert_threshold = 0.5
    prediction = detect_triggerword(file_name, model)
    action_on_activate(chime_file, file_name, prediction, alert_threshold)

if __name__== "__main__":
    main()
