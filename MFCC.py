# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 20:33:48 2022

@author: shubh
"""

import numpy, scipy, matplotlib.pyplot as plt, sklearn, librosa, urllib, IPython.display,os
import librosa
import matplotlib.pyplot as plt
import librosa.display
#plt.rcParams['figure.figsize'] = (10,4)

for i in range(815):
    path = r"C:\Users\shubh\Downloads\Trim 3\\"
    di="C:/Users/shubh/Downloads/MFCC"
    scale_file = os.listdir(path)
    o=path+scale_file[i]
    print(o)
    plt.rcParams['figure.figsize'] = (10,4)
    x, fs = librosa.load(o)

    mfccs1 = librosa.feature.mfcc(x, sr=fs)

    mfccs = sklearn.preprocessing.scale(mfccs1, axis=1)
    #print (mfccs.mean(axis=1))
    #print (mfccs.var(axis=1))

    librosa.display.specshow(mfccs, sr=fs, x_axis='time')

    name = scale_file[i].split(".")[0]
    
    plt.savefig(di+"\\"+name+".png",dpi=150,format="png")
    