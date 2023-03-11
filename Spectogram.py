# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:02:54 2022

@author: shubh
"""
import os
import glob
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt


for i in range (2):
    path = "C:/Users/shubh/Downloads/archive (1)/train/train/"
    di="C:/Users/shubh/Downloads/archive (1)/train/specto"
    scale_file =  os.listdir(path)
    
    o=path+scale_file[i]
    print(i)
    

#ipd.Audio(scale_file[0])
# load audio files with librosa
#scale, sr = librosa.load(path,scale_file[0])

    plt.rcParams["figure.figsize"] = [12, 6]
    plt.rcParams["figure.autolayout"] = True

    fig, ax = plt.subplots()

    hl = 512 # number of samples per time-step in spectrogram
    hi = 224 # Height of image
    wi = 224 # Width of image

# Loading demo track
    y, sr = librosa.load(o)
    window = y[0:wi*hl]


    S = librosa.feature.melspectrogram(y=window, sr=sr, n_mels=hi, fmax=8000,hop_length=hl)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
    
    #name = o.split(".")[0]
    name = scale_file[i].split(".")[0]
    
    plt.savefig(di+"\\"+name+"1"+".png",dpi=150,format="png")
   