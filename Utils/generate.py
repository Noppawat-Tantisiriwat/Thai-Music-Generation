import tensorflow as tf
import librosa
import soundfile as sf
import pickle as p
import numpy as np
import argparse
import os
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("-n", "--num_generate", type=int, default=10, help="number of files generated")

parser.add_argument("-m", "--model", type=str, default=None, help="enter vae model path")

parser.add_argument("-o", "--output", type=str, default=os.getcwd(), help="enter vae model path")

args = parser.parse_args()

num_generate = args.num_generate
model_path = args.model
output = args.output



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

def main(num_generate, model_path, output):
    os.mkdir(output)
    vae = tf.keras.models.load_model(model_path)
    for i in tqdm(range(num_generate)):
        wave = generate(vae)
        sf.write(os.path.join(output, f"generation_no.{i+1:02d}.wav"), wave, samplerate=22050)




if __name__ == '__main__':
    main(num_generate, model_path, output)