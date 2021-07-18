import os
import matplotlib.pyplot as plt

import librosa
import librosa.display

import IPython.display as ipd
from PIL import Image

# from __future__ import unicode_literals
import youtube_dl
import pandas as pd
import numpy as np
from scipy import signal
from scipy.io import wavfile

audio_path = "../../data/musiques/"
spec_path = "../../data/spectrogrammes/"
music_name = audio_path + "tmp_file"
csv_path = "../../data/csv_files.csv"

def convertYtb(url_name):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': music_name + '.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192'
        }],
        'postprocessor_args': [
            '-ar', '16000'
        ],
        'prefer_ffmpeg': True
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url_name])

def import_images_and_resize(file):
    image_path = os.path.join(spec_path, file)
    im = Image.open(image_path)
    im = im.resize((8, 8))
    im = im.convert("RGB")
    im_arr = np.array(im)
    im_arr = np.reshape(im_arr, (8 * 8 * 3))

    return im_arr

def convertSpect(row_number, music_type):
    audio_clips = os.listdir(audio_path)
    
    x, sr = librosa.load(audio_path+audio_clips[0], sr=441000,offset=60.0, duration=15.0) 
    
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14,5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()

    path_img = spec_path + "music_" + str(music_type) + "_" + str(row_number) 
    plt.savefig(path_img+".png")
    plt.close('all')

def deletePicture():
    music = music_name+".wav"
    os.remove(music)

    picture = os.path.join(spec_path, "music_1_test.png")
    os.remove(picture)