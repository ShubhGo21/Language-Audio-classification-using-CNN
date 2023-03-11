# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 07:54:34 2022

@author: shubh
"""
import os
from pydub import AudioSegment

for i in range (1):
    files_path = "C:/Users/shubh/Downloads/aud.wav"
    di="C:/Users/shubh/Desktop"
    file_name = os.listdir(files_path)
    o = files_path+file_name[i]
    print(o)
    
    
    startMin = 0
    startSec = 0
    
    endMin = 0
    endSec = 3
    
    # Time to miliseconds
    startTime = startMin*60*1000+startSec*1000
    endTime = endMin*60*1000+endSec*1000
    
    # Opening file and extracting segment
    song = AudioSegment.from_wav(o)
    extract = song[startTime:endTime]
    
    # Saving
    # extract.export( files_path+file_name+'_extract.mp3', format="mp3")
    name=file_name[i].split(".")[0]
    extract.export(di+"\\"+name+"1"+".wav", format="wav")