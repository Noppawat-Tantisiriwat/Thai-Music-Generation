from typing import List
from tensorflow import keras
import librosa
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv1D, LayerNormalization, \
    Flatten, Dense, Reshape, Conv1DTranspose, Layer
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tqdm.auto import tqdm
import numpy as np
import datetime, os
import pickle as p
import urllib.request
from google_drive_downloader import GoogleDriveDownloader as gdd
import streamlit as st
import soundfile as sf


on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    html_theme = 'default'
else:
    html_theme = 'nature'
MODEL_WEIRDO = "10QZqFAFkD5rRwr-sXPHttLxsRgfZ6hjH"

# MODEL_CNN = "https://drive.google.com/drive/folders/1WgKXGMJgaMgBSDNLPFmtUhIX3j-Ywcoa?usp=sharing"

# MODEL_CNN_LSTM = "https://drive.google.com/drive/folders/15MLS-UF-njBEYx2Qg1EoSzSZWJbaAOoy?usp=sharing"
gdd.download_file_from_google_drive(file_id=MODEL_WEIRDO,
                                    dest_path="weirdo/variables/variables.data-00000-of-00001"
                                    )


# urllib.request.urlretrieve(MODEL_WEIRDO, "cnn")
# urllib.request.urlretrieve(MODEL_WEIRDO, "cnn_lstm")

# cnn = keras.models.load_model("cnn")
# cnn_lstm = keras.models.load_model("cnn_lstm")
weirdo = keras.models.load_model("weirdo")

min_max = "1oGk9dnPOSCPUXskTTs6uWElnSZipCgk6"

gdd.download_file_from_google_drive(file_id=min_max,
                                    dest_path="./min_max_values.pkl"
                                    )


with open("min_max_values.pkl", "rb") as file:
  max_min = p.load(file)
min_li = []
max_li = []
for _, value in max_min.items():
    max_li.append(value["max"])
    min_li.append(value["min"])
min_array = np.array(min_li)
max_array = np.array(max_li)
min_original = np.mean(min_array)
max_original = np.mean(max_array)


def denormalize(array, min_original, max_original):
    array = (array - 0.) / (1. - 0.)
    array = array * (max_original - min_original) + min_original
    return array

def generate(model):
    eps = tf.random.normal([1, 1024])
    log_spectrogram = model.decoder(eps)
    log_spectrogram = tf.squeeze(log_spectrogram).numpy().T
    log_denorm = denormalize(log_spectrogram, min_original, max_original)
    spectrogram = librosa.db_to_amplitude(log_denorm)
    wave = librosa.griffinlim(spectrogram, hop_length=256, win_length=510)
    return wave
        
        
st.title("Thai Music (Ranat Ek) Generatation with VAE")
st.sidebar.write("It's time to generate!!!")
if st.sidebar.button("Generate", help="press to generate music"):
    wave = generate(weirdo)
    sf.write("test.wav", wave, samplerate=22050)
    audio_file = open('test.wav', 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes)






